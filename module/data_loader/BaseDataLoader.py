import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from collections import namedtuple
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import random
import cv2


class BaseDataLoader(DataLoader):
    _args = {
        "batch_size": 16,
        "num_workers": 1,
        "imgs_mean": (0.5, 0.5, 0.5),  # imagenet
        "imgs_std": (1, 1, 1),  # imagenet
        "training": True,
        "split": 0,
        "test_mode": False,
        "collate_fn": default_collate,
        "resize": None,
        "masks_col": 'masks',
        'img_col': 'images',
        'data_augement': False
    }

    def __init__(self, datasets, valid_datasets=None, *args, **kwargs):
        self._args.update(kwargs)
        self.dataset = datasets
        self.valid_datasets = valid_datasets

        self.n_samples = len(self.dataset)
        print("get %d data for train_data" % (self.n_samples))

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self._args['batch_size'],
            'shuffle': self._args['training'],
            'collate_fn': self._args['collate_fn'],
            'num_workers': self._args['num_workers'],
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)

    def __init_valid_dataloader(self):
        init_kwargs = {
            'dataset': self.valid_datasets,
            'batch_size': self._args['batch_size'],
            'shuffle': self._args['training'],
            'collate_fn': self._args['collate_fn'],
            'num_workers': self._args['num_workers'],
            'pin_memory': True
        }
        return DataLoader(**init_kwargs)

    def split_validation(self):
        if self._args['test_mode']:
            return self
        else:
            return self.__init_valid_dataloader()
