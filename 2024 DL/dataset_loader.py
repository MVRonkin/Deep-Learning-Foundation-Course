"""
author: Praveen Perumal
datetime: 29 May 2024 at 1:06 PM
"""

import struct, array
import numpy as np
import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, images_filepath, labels_filepath, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.images = open(images_filepath, 'rb')
        self.labels = open(labels_filepath, 'rb')

        magic, size = struct.unpack(">II", self.labels.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        self.size = size

        magic, size, rows, cols = struct.unpack(">IIII", self.images.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        self.rows = rows
        self.cols = cols

    def __del__(self):
        self.images.close()
        self.labels.close()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self.labels.seek(8 + idx)
        label, = struct.unpack(">B", self.labels.read(1))
        label = torch.tensor(label)

        self.images.seek(16 + idx * self.rows * self.cols)
        image = array.array("B", self.images.read(self.rows * self.cols))
        image = np.asarray(image, dtype=float).reshape(self.rows, self.cols)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image.float(), label.long()

# transform = lambda x: x.flatten() / 255.0
# target_transform = lambda x: torch.zeros(10).scatter_(0, x, value=1)

# training_data = MNISTDataset(images_filepath="dataset/train-images.idx3-ubyte", 
#                              labels_filepath="dataset/train-labels.idx1-ubyte",
#                              transform=transform,
#                              target_transform=target_transform)
# test_data = MNISTDataset(images_filepath="dataset/t10k-images.idx3-ubyte",
#                          labels_filepath="dataset/t10k-labels.idx1-ubyte",
#                          transform=transform,
#                          target_transform=target_transform)
