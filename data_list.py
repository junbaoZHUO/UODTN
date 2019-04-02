#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
import os
import os.path
import time

def make_dataset(image_list, labels):
    if len(image_list[0].split()) > 2:
      images = [(val.split()[0], int(val.split()[1]), val.split()[2], int(val.split()[3])) for val in image_list]
    else:
      images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
        return pil_loader(path)


class ImageList(object):

    def __init__(self, image_list, shape=None,labels=None, transform=None, target_transform=None,
                 loader=default_loader, train=True):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.shape = shape#hassassin
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.train:
              path, target, path2, target2 = self.imgs[index]
        else:
              path, target = self.imgs[index]
        img = self.loader(path)
        if self.train:
            img2 = self.loader(path2)
        if self.transform is not None:
            img = self.transform(img)
            if self.train:
                img2 = self.transform(img2)
        if self.train:
            return img, target, target2, img2
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

def ClassSamplingImageList(image_list, transform, return_keys=False):
    data = open(image_list).readlines()
    label_dict = {}
    for line in data:
        label_dict[int(line.split()[1])] = []
    for line in data:
        label_dict[int(line.split()[1])].append(line)
    all_image_list = {}
    for i in label_dict.keys():
        all_image_list[i] = ImageList(label_dict[i], transform=transform)
    if return_keys:
        return all_image_list, label_dict.keys()
    else:
        return all_image_list
