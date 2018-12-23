# coding: utf-8
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
from PIL import Image


class MNIST_data(Dataset):
    """MNIST dataset"""

    def __init__(self, file_path, n_pixels,
                 transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))])
                 ):

        df = pd.read_csv(file_path)
        if len(df.columns) == n_pixels + 1:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)

        else:
            # test data
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = None
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])


class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift

    @staticmethod
    def get_params(shift):
        hshift, vshift = np.random.uniform(-shift, shift, size=2)
        return hshift, vshift

    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift), resample=Image.BICUBIC, fill=1)
