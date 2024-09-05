import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import seed
from torch.optim import lr_scheduler, optimizer
import utils

from dataloaders.HEDataloader import HEDataModule
from models import helper, regression
import numpy as np
from prettytable import PrettyTable
# from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.loggers import CSVLogger


class HEModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        # self.miner_name = miner_name
        # self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu

        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        self.regressor = regression.Regression(in_dim=5120, regression_ratio=0.8)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        h = self.regressor(x)
        return h
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, preds, gt):
        # we mine the pairs/triplets if there is an online mining strategy
        loss = self.loss_fn(preds, gt)
        batch_acc = 0.0
        if type(loss) == tuple: 
            # somes losses do the online mining inside (they don't need a miner objet), 
            # so they return the loss and the batch accuracy
            # for example, if you are developping a new loss function, you might be better
            # doing the online mining strategy inside the forward function of the loss class, 
            # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
            loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, heights = batch

        # Feed forward the batch to the model
        pred_heights = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(pred_heights, heights) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        self.training_step_outputs.append(loss)
        
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_mean", epoch_mean)
        # we empty the batch_acc list for next epoch
        self.batch_acc = []
        self.training_step_outputs.clear()

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, gt = batch
        # calculate descriptors
        heights = self(places)
        self.validation_step_outputs.append(heights.detach().cpu())
        loss = self.loss_function(heights, gt)
        self.log('val_loss', loss.item(), logger=True)
        return loss
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)

        val_step_outputs = self.validation_step_outputs
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            gt_heights = val_dataset.heights
            range_threshold = [25, 50, 75, 100, 125, 150]
            count_true_list = []
            for k in range_threshold:
                count_true = 0
                for i in range(len(gt_heights)):
                    if abs(gt_heights[i] - feats[i]) < k:
                        count_true += 1
                count_true_list.append(count_true)
            true_percentage = np.array(count_true_list) / len(gt_heights)
            recall_dict = {k:v for (k,v) in zip(range_threshold, true_percentage)}
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in range_threshold]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in true_percentage])
            print(table.get_string(title=f"Performances on {val_set_name}"))

                
            for k in range_threshold:
                self.log(f'{val_set_name}/R{k}', recall_dict[k], prog_bar=False, logger=True)
        print('\n\n')


if __name__ == '__main__':
    
    seed.isolate_rng(include_cuda=True)
        
    datamodule = HEDataModule(
        batch_size=32,
        image_size=(360, 480),
        num_workers = 16,
        show_data_stats=True,
        random_sample_from_each_place=True,
        train_set_names=['2022','2020','2017','2013','2019'],
        val_set_names=['real_photo'], 
    )
    
    # examples of backbones
    # resnet18, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
    # efficientnet_b0, efficientnet_b1, efficientnet_b2
    # swinv2_base_window12to16_192to256_22kft1k
    model = HEModel(
        #---- Encoder
        backbone_arch='efficientnet_v2_m',
        pretrained=True,
        layers_to_freeze=2,
        # layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        layers_to_crop=[4],
        
        #---- Aggregator
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 2048,
        #             'out_dim': 2048},
        # agg_arch='GeM',
        # agg_config={'p': 3},
        
        # agg_arch='ConvAP',
        # agg_config={'in_channels': 2048,
        #             'out_channels': 2048},

        agg_arch='MixVPR',
        agg_config={'in_channels' : 1280,
                'in_h' : 12,
                'in_w' : 15,
                'out_channels' : 1280,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4,
                }, # the output dim will be (out_rows * out_channels)
        
        #---- Train hyperparameters
        lr=0.05, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
        optimizer='sgd', # sgd, adamw
        weight_decay=0.001, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=650,
        milestones=[5, 10, 15, 25, 45],
        lr_mult=0.3,

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='Huber',
        faiss_gpu=False
    )
    
    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = ModelCheckpoint(
        monitor='real_photo/R25',
        filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_R25[{real_photo/R25:.4f}]_R50[{real_photo/R50:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=5,
        mode='max',)
    
    checkpoint_val_loss = ModelCheckpoint(
        monitor='val_loss',
        filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_R25[{real_photo/R25:.4f}]_R50[{real_photo/R50:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',)
    
    checkpoint_train_loss = ModelCheckpoint(
        monitor='loss',
        filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_R25[{real_photo/R25:.4f}]_R50[{real_photo/R50:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=5,
        mode='max',)
    
    logger = CSVLogger("logs", name="my_exp_name")

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu', devices=[0],
        default_root_dir=f'./LOGS/{model.encoder_arch}', # Tensorflow can be used to viz 
        # logger=CSVLogger,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=16, # we use half precision to reduce  memory usage
        max_epochs=80,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb, checkpoint_val_loss, checkpoint_train_loss],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=10,
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
