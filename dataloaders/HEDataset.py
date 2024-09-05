# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from glob import glob
import numpy as np


default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = '/root/shared-storage/shaoxingyu/workspace_backup/gsvqddb_train/'
real_BASE_PATH = '/root/shared-storage/shaoxingyu/workspace_backup/dcqddb_test/'


if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

default_transform = T.Compose([
    # T.Resize(args.train_resize, antialias=True),
    # T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1], antialias=True),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

basic_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class HEDataset(Dataset):
    def __init__(self,
                 foldernames=['2013', '2017', '2019', '2020', '2022'],
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH
                 ):
        super(HEDataset, self).__init__()
        self.base_path = base_path
        self.foldernames = foldernames

        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframes = self.__getdataframes()
        self.heights = list(self.dataframes['flight_height'].values)
        self.year = list(self.dataframes['year'])
        self.flight_height = list(self.dataframes['flight_height'])
        self.alpha = list(self.dataframes['alpha'])
        self.loc_x = list(self.dataframes['loc_x'])
        self.loc_y = list(self.dataframes['loc_y'])
        
        # get all unique place ids
        self.total_nb_images = len(self.dataframes)
        
    def __getdataframes(self) -> pd.DataFrame:
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path + 'Dataframes/'+f'{self.foldernames[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.foldernames)):
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.foldernames[i]}.csv')

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        if self.random_sample_from_each_place:
            df = df.sample(frac=1)

        # keep only places depicted by at least min_img_per_place images
        # res = df[df.groupby('place_id')['place_id'].transform('size') >= self.min_img_per_place]
        # return res.set_index('place_id')

        return df.reset_index(drop=True)
    
    def __getitem__(self, index):
        
        height = self.heights[index]
        image_name = f'@{self.year[index]}@{self.flight_height[index]:.2f}@{self.alpha[index]:.2f}@{self.loc_x[index]}@{self.loc_y[index]}@.png'
        # f'{year}@{flight_height}@{alpha}@{loc_w}@{loc_h}.png'
        image_path = self.base_path + 'Images/' + image_name
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, height
    
    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.heights)



class realHEDataset(Dataset):
    def __init__(self, base_path=real_BASE_PATH, transform=basic_transform):
        super().__init__()
        self.base_path = base_path
        self.transform = transform
        images_paths = sorted(glob(f"{real_BASE_PATH}/**/*.png", recursive=True))
        
        self.images_paths = images_paths
        self.heights = self.get_heights(images_paths)

    def __getitem__(self, index):
        
        height = self.heights[index]
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, height
    
    def __len__(self):
        return len(self.images_paths)
    
    @staticmethod
    def get_heights(image_paths):
        info_list = [image_path.split('/\\')[-1].split('@') for image_path in image_paths]
        heights = np.array([info[-4] for info in info_list]).astype(np.float64)
        return heights



        
    