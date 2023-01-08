# -*- coding: utf-8 -*-

import os, sys, pdb
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
import os.path
import os, sys, pdb
import torch
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
print(f"loaded {__name__}")

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

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

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

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

        return img, target#, filename

    def __len__(self):
        return len(self.imgs)


def create_datasets(data_dir, phases = ['train', 'val', 'test', 'auto_test']):
    pixel_mean, pixel_std = 0.66133188,  0.21229856
    data_transform = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ])
    }

    dsets = {x: ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in phases}
    return dsets


def load_multi_gpu(dsets, batch_size):
    out = {}
    for phase, dataset in dsets.items():
        out[phase] = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False, 
            #num_workers=4, 
            pin_memory=True,
            sampler=DistributedSampler(dataset)
            )
    return out


def load_single_gpu(dsets, batch_size):
    out = {}
    for phase, dataset in dsets.items():
        out[phase] = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=(phase=="train"), 
            pin_memory=False,
            num_workers = 4
            )
    return out




def data_load(data_dir, batch_size):
    pixel_mean, pixel_std = 0.66133188,  0.21229856
    phases = ['train', 'val', 'test', 'auto_test']
    # phases = ['train', 'val', 'test', 'auto_test']
    data_transform = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ])
    }

    dsets = {x: ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in phases}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
            shuffle=(x=='train'), num_workers=4) for x in phases}
    dset_classes = dsets['train'].classes
    dset_size = {x: len(dsets[x]) for x in phases}
    num_class = len(dset_classes)

    return dset_loaders, dset_size, num_class