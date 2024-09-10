import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms as T
import logging
from datetime import datetime
import sys
import torchmetrics
from tqdm import tqdm
from math import sqrt

from dataloaders.HEDataset import HEDataset, realHEDataset
from models import helper, regression
import commons

from utils.checkpoint import save_checkpoint

train_batch_size = 32
num_workers = 16
num_epochs = 100

scheduler_patience = 10
lr = 0.00001

seed = 0

resume_train = False

device = "cuda" if torch.cuda.is_available() else "cpu"

foldernames=['2013', '2017', '2019', '2020', '2022', 'real_photo']
train_dataset_folders = ['2013', '2017', '2019', '2020', '2022']
test_datasets = ['real_photo']

image_size = (360, 480)

range_threshold = [25, 50, 75, 100, 125, 150]

# backbone_arch = 'efficientnet_v2_m'
# agg_arch='MixVPR'
# agg_config={'in_channels' : 1280,
#             'in_h' : 12,
#             'in_w' : 15,
#             'out_channels' : 640,
#             'mix_depth' : 4,
#             'mlp_ratio' : 1,
#             'out_rows' : 4,
#             }   # the output dim will be (out_rows * out_channels)


backbone_arch = 'dinov2_vitb14'
agg_arch='MixVPR'
agg_config={'in_channels' : 768,
            'in_h' : 25,
            'in_w' : 34,
            'out_channels' : 384,
            'mix_depth' : 4,
            'mlp_ratio' : 1,
            'out_rows' : 4,
            }   # the output dim will be (out_rows * out_channels)


regression_in_dim = agg_config['out_rows'] * agg_config['out_channels']




exp_name = f'HE-{backbone_arch}-{agg_arch}'
save_dir = os.path.join("logs", exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


train_transform = T.Compose([
    T.Resize(image_size, antialias=True),
    # T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1], antialias=True),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform =T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#### 初始化
commons.make_deterministic(seed)
commons.setup_logging(save_dir, console="info")
logging.info(" ".join(sys.argv))
# logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {save_dir}")

#### Dataset & Dataloader
train_dataset = HEDataset(train_dataset_folders, random_sample_from_each_place=True,transform=train_transform)
train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
iterations_num = len(train_dataset) // train_batch_size


test_dataset_list = []
test_datasets_load = test_datasets
if 'real_photo' in test_datasets:
    real_photo_dataset = realHEDataset()
    test_datasets_load.remove('real_photo')
    test_dataset_list.append(real_photo_dataset)
if len(test_datasets_load) != 0:
    fake_photo_dataset = HEDataset(foldernames=test_datasets_load, random_sample_from_each_place=False,transform=test_transform)
    test_dataset_list.append(fake_photo_dataset)
if len(test_dataset_list) > 1:
    test_dataset = ConcatDataset(test_dataset_list)
else:
    test_dataset = test_dataset_list[0]
test_img_num = len(test_dataset)
test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

#### model
# backbone = helper.get_backbone(backbone_arch=backbone_arch, pretrained=True, layers_to_freeze=9, layers_to_crop=[])
# aggregator = helper.get_aggregator(agg_arch=agg_arch, agg_config=agg_config)
# # regressor = regression.Regression(in_dim=1024, regression_ratio=0.5)
# regressor = regression.Regression(in_dim=regression_in_dim, regression_ratio=0.5)

backbone = helper.get_backbone(backbone_arch=backbone_arch, num_trainable_blocks=2)
aggregator = helper.get_aggregator(agg_arch=agg_arch, agg_config=agg_config)
regressor = regression.Regression(in_dim=regression_in_dim, regression_ratio=0.5)

backbone = backbone.to(device)
aggregator = aggregator.to(device)
regressor = regressor.to(device)

model = nn.Sequential(backbone, aggregator, regressor)
# model = nn.Sequential(backbone, aggregator)

#### OPTIMIZER & SCHEDULER
optimizer = optim.Adam(model.parameters(), lr=lr)
# regression_optimizer = optim.Adam(regressor.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True) #NOTE: 学习率变化

#### Resume
# if resume_train:
#     model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num = \
#         util.resume_train_with_groups(args, args.save_dir, model, optimizer, classifiers, classifiers_optimizers)
#     epoch_num = start_epoch_num - 1
#     best_loss = best_train_loss
#     logging.info(f"Resuming from epoch {start_epoch_num} with best train loss {best_train_loss:.2f} " +
#                  f"from checkpoint {args.resume_train}")


best_loss = 100
### Train&Loss



# 初始化模型、损失函数和优化器
criterion = nn.MSELoss()
scaler = torch.GradScaler(device)
# cross_entropy_loss = nn.CrossEntropyLoss()


# 训练模型
for epoch in range(num_epochs):
    if optimizer.param_groups[0]['lr'] < 1e-6:
        logging.info('LR dropped below 1e-6, stopping training...')
        break

    train_loss = torchmetrics.MeanMetric().to(device)
    train_acc = torchmetrics.Accuracy(task='binary').to(device)


    
    model.train()
    tqdm_bar = tqdm(range(iterations_num), ncols=100, desc="")
    iteration = 0
    # for iteration in tqdm_bar:
    for images, heights_gt in train_dl:
        iteration += 1
        # images, heights_gt = train_dl
        images, heights_gt = images.to(device), heights_gt.to(device)
        optimizer.zero_grad()
        # regression_optimizer.zero_grad()
        with torch.autocast(device):
            # disctiptors = model(images)
            # heights_pred = regressor(disctiptors)
            # loss = cross_entropy_loss(disctiptors, heights_gt)

            if iteration > 155:
                pass
            heights_pred = model(images)
            loss = criterion(heights_pred, heights_gt)
            # loss = loss / train_batch_size
            # loss = sqrt(loss)
            
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # train_acc.update(heights_pred, heights_gt)
        train_loss.update(loss.item())
        tqdm_bar.set_description(f"{loss.item():.1f}")
        # tqdm_bar.n = (100*iteration) // iterations_num
        _ = tqdm_bar.refresh()
        _ = tqdm_bar.update()

    #### Validation
    model.eval()

    count_valid_recall = torch.zeros(len(range_threshold))
    valid_recall_percentage = torch.zeros(len(range_threshold))
    range_threshold_tensor = torch.tensor(range_threshold)

    valid_heigths = torch.zeros(test_img_num, len(range_threshold),dtype=torch.bool)

    with torch.no_grad():
        # tqdm_bar = tqdm(range(), ncols=100, desc="")
        # for images, heights_gt in test_dl:
        for query_i, (images,heights_gt) in enumerate(test_dl):
            images = images.to(device)
            heights_gt = torch.tensor(heights_gt).to(device)
            recall_heights_range = torch.zeros(test_img_num, len(range_threshold))
            # with torch.autocast(device):
            heights_pred = model(images)
            distances = abs(heights_pred - heights_gt)
            range_threshold_tensor = range_threshold_tensor.to(distances.device)
            valid_heigths[query_i,:] = distances < range_threshold_tensor
        count_valid_recall = torch.count_nonzero(valid_heigths, dim=0)
        valid_recall_percentage = count_valid_recall / test_img_num

    val_recall_str = ", ".join([f'LR@{N}: {acc:.1f}' for N, acc in zip(range_threshold, valid_recall_percentage)])

    # train_acc = train_acc.compute() * 100
    train_loss = train_loss.compute()

    if train_loss < best_loss:
        is_best = True
        best_loss = train_loss
    else:
        is_best = False

    logging.info(f"E{epoch: 3d}, " + 
                #  f"train_acc: {train_acc.item():.1f}, " +
                 f"train_loss: {train_loss.item():.2f}, best_train_loss: {scheduler.best:.2f}, " +
                 f"not improved for {scheduler.num_bad_epochs}/{scheduler_patience} epochs, " +
                 f"lr: {round(optimizer.param_groups[0]['lr'], 21)}")
    logging.info(f"E{epoch: 3d}, Val LR: {val_recall_str}") # NOTE 测试召回率？

    scheduler.step(train_loss)
    
    save_checkpoint({"epoch_num": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "classifiers_state_dict": [c.state_dict() for c in classifiers],
        # "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        # "args": args,
        "best_train_loss": best_loss
    }, is_best, save_dir)


print("Training complete.")