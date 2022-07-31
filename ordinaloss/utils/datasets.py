# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:44:20 2022

@author: imargolin
"""

from skimage import io
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
import os


class AgeDataSetOnMemory(Dataset):
    def __init__(self, df, transform=None):
        #Heavy in memory
        self.df = df
        self.idx_to_cat = df["age_category"].cat.categories
        self.cat_to_idx = {cat:idx for idx, cat in enumerate(self.idx_to_cat)}
        self.transform = transform
        
        #HEAVY!
        imgs = df["image_path"].apply(io.imread).apply(transform).tolist()
        self.imgs = torch.stack(imgs)
        
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.df["age_category"].iloc[idx]
        label = self.cat_to_idx[label]
        
        return img, label
    
    def __len__(self):
        return len(self.df)
    
class AgeDataset(Dataset):
    def __init__(self, df, device, transform=None):
        self.df = df
        self.idx_to_cat = df["age_category"].cat.categories
        self.cat_to_idx = {cat:idx for idx, cat in enumerate(self.idx_to_cat)}
        self.transform = transform
        
    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        
        image = io.imread(info["image_path"])
        label = self.cat_to_idx[info["age_category"]]
        if self.transform:
            image = self.transform(image) 
            
        return image, label
    
    def __len__(self):
        return len(self.df)
    
    def visualize_one(self, idx):
        img, label = self[idx]
        img = img.permute(1,2,0).detach().numpy()
        plt.imshow(img)
        plt.title(f"label: {self.idx_to_cat[label]}")
        
#TODO
class ImageFolder(Dataset):
    
    
    def __init__(self, root, transform=None, target_transform=None):
        '''
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
        '''
        
        
        #classes, class_to_idx = find_classes(root)
        #imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        #self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = os.path.basename(path)

        return img, target, filename

    def __len__(self):
        return len(self.imgs)

