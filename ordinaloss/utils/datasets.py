# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:44:20 2022

@author: imargolin
"""

from skimage import io
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
import os

def create_metadata_of_imgs(root: str) -> pd.DataFrame:
    out = []
    for label_path in Path(root).iterdir():
        for img in label_path.iterdir():
            out.append((label_path.name, img.as_posix()))
    
    return pd.DataFrame(out, columns = ["label", "full_path"])


def load_img(path):    
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


def create_transform_pipeline(brightness = 0, 
                              contrast = 0,
                              saturation = 0, 
                              hue=0):
    pixel_mean, pixel_std = 0.66133188, 0.21229856
    
    transform_pipeline = transforms.Compose([
    transforms.ColorJitter(brightness, contrast, saturation, hue),
    transforms.ToTensor(),
    transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ])
    return transform_pipeline


class ImageFolder(Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None):
        
        self.root = root
        self.imgs = create_metadata_of_imgs(self.root)
        
        labels = self.imgs["label"].unique().tolist()
        self.target_to_idx = dict(zip(labels, range(len(labels))))
        self.idx_to_target = {v:k for k,v in self.target_to_idx.items()}

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        target, path = self.imgs.iloc[index]
        target_idx = self.target_to_idx[target]
        
        img = load_img(path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        #filename = os.path.basename(path)

        return img, target_idx

    def __len__(self):
        return len(self.imgs)
