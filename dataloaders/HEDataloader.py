import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.HEDataset import HEDataset, realHEDataset

from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

default_transform = T.Compose([
    # T.Resize(args.train_resize, antialias=True),
    # T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1], antialias=True),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class HEDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 image_size=(360, 480),
                 num_workers=4,
                 show_data_stats=True,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 random_sample_from_each_place=True,
                 train_set_names = ['pitts30k_val', 'msls_val'],
                 val_set_names=['pitts30k_val', 'msls_val'],
                #  use_real_photo = False
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.show_data_stats = show_data_stats
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_place = random_sample_from_each_place
        self.train_set_names = train_set_names
        self.val_set_names = val_set_names
        # self.use_real_photo = use_real_photo
        self.use_real_photo = 'real_photo' in self.val_set_names
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        # self.train_transform = T.Compose([
        #     T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        #     T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        #     T.ToTensor(),
        #     T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        # ])
        
        self.train_transform = default_transform

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.random_sample_from_each_place}

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader with reload routine
            self.reload()

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            if self.use_real_photo:
                real_photo_dataset = realHEDataset()
                self.val_datasets.append(real_photo_dataset)
            if len(self.val_set_names)-int(self.use_real_photo) != 0:
                val_dataset = HEDataset(foldernames=self.val_set_names, random_sample_from_each_place=False,transform=self.valid_transform)
                self.val_datasets.append(val_dataset)

                    # print(
                    #     f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                    # raise NotImplementedError
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = HEDataset(foldernames=self.train_set_names, random_sample_from_each_place=True, transform=self.train_transform)

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of folders", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        if self.use_real_photo:
            table.add_row([f"Validation set use real photo", "True"])
        # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size (PxK)", f"{self.batch_size}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
