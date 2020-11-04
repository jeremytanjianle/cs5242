"""
In order to generate train and validation data set easily, 
we create a custom dataset object for pytorch
"""
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
import numpy as np
import random

import tarfile
import io
import os
import pandas as pd

from torch.utils.data import Dataset
import torch


class cs5242_dataset(Dataset):
    def __init__(self, txt_path='data/nus-cs5242/train_label.csv', 
                 img_dir='data/nus-cs5242/train_image/train_image', 
                 transform=None, 
                 train=True, split=0.8
                ):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        :param train: boolean, indicating which way to split
        :param split: float less than 1, indicating how to split the data
        """
        # init labels and split
        self.df = pd.read_csv(txt_path, index_col=0)
        # The augmented data is flushed the back
        # so we put the split in the front 
        # to keep the augmented data in the training set
        self.df = self.df.iloc[-int(split*len(self.df)):] if train else self.df.iloc[:-int(split*len(self.df))] 
        
        # dir values for images
        self.img_names = self.df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        
        # transform settings
        self.transform = transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        self.get_image_selector = True if img_dir.__contains__('tar') else False
        self.tf = tarfile.open(self.img_dir) if self.get_image_selector else None

    def get_image_from_tar(self, name):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        image = self.tf.extractfile(name)
        image = image.read()
        image = Image.open(io.BytesIO(image))
        return image

    def get_image_from_folder(self, name):
        """
        gets a image by a name gathered from file list text file

        :param name: name of targeted image
        :return: a PIL image
        """

        image = Image.open(os.path.join(self.img_dir, str(name) + '.png'))
        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        if index == (self.__len__() - 1) and self.get_image_selector:  # close tarfile opened in __init__
            self.tf.close()

        if self.get_image_selector:  # note: we prefer to extract then process!
            X = self.get_image_from_tar(self.img_names[index])
        else:
            X = self.get_image_from_folder(self.img_names[index])

        Y = self.df.loc[self.img_names[index],'Label'] # torch.tensor(df.loc[self.img_names[index]].values.flatten()) # df.loc[self.img_names[index]] 
        
        if self.transform is not None:
            X = self.transform(X)

        return X, Y