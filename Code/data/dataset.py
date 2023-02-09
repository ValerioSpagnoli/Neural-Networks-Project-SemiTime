import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

from augmentation import magnitude_wrap, time_wrap


class LabelledDataset(Dataset):
    def __init__(self, dataframe=None, augmentation=None):
        self.df = dataframe
        self.augmentation = augmentation
  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = np.array(self.df['target'][idx])
        features = np.array(self.df.iloc[idx:idx+1, :len(self.df.columns)-1])
        features = features.reshape(features.shape[1], features.shape[0])

        target = torch.from_numpy(target).to(torch.int64)
        features = torch.from_numpy(features).to(torch.float32)

        if self.augmentation == True:
            mw = magnitude_wrap.MagnitudeWrap(sigma=0.3, knot=4)
            tw = time_wrap.TimeWarp(sigma=0.3, knot=4)
            
            transform_mw = transforms.Compose([mw]) 
            transform_tw = transforms.Compose([tw])
            
            features_mw = torch.Tensor(transform_mw(features)).to(torch.float32)
            features_tw = torch.Tensor(transform_tw(features)).to(torch.float32)

            features = torch.cat((features, features_mw, features_tw))
            return features, target

        return  features, target
    


class UnlabelledDataset(Dataset):
    def __init__(self, dataframe=None, augmentation=None, split_ratio=None):
        self.df = dataframe
        self.augmentation = augmentation
        self.split_ratio = split_ratio

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        target = np.array(self.df['target'][idx])
        features = np.array(self.df.iloc[idx:idx+1, :len(self.df.columns)-1])
        features = features.reshape(features.shape[1], features.shape[0])

        target = torch.from_numpy(target).to(torch.int64)
        features = torch.from_numpy(features).to(torch.float32)

        if self.augmentation == True:
            mw = magnitude_wrap.MagnitudeWrap(sigma=0.3, knot=4)
            tw = time_wrap.TimeWarp(sigma=0.3, knot=4)
            
            transform_mw = transforms.Compose([mw]) 
            transform_tw = transforms.Compose([tw])
            
            features_mw = torch.Tensor(transform_mw(features)).to(torch.float32)
            features_tw = torch.Tensor(transform_tw(features)).to(torch.float32)

            features = torch.cat((features, features_mw, features_tw), dim=0)

        past_size = int(np.ceil(features.shape[0]*self.split_ratio))
        future_size = int(np.floor(features.shape[0]*(1-self.split_ratio)))

        self.features_past, self.features_future = torch.split(features, [past_size, future_size])        

        return  self.features_past, self.features_future