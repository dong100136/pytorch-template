import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations import (Compose, HorizontalFlip, RandomBrightness,
                            RandomContrast, ShiftScaleRotate)
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from ...registry import DATA_LOADER
from ..BaseDataLoader import BaseDataLoader


@DATA_LOADER.register("TgsDataLoader")
class TgsDataLoader(BaseDataLoader):
    def __init__(self, train_csv, valid_csv, *args, **kwargs):
        self.test_mode = True

        self.train_dataset = CSVImgDataSet(train_csv)
        if self.test_mode:
            self.valid_dataset = self.train_dataset
        else:
            self.valid_dataset = CSVImgDataSet(valid_csv)

        return super().__init__(self.train_dataset, self.valid_dataset, *args, **kwargs)


class CSVImgDataSet(Dataset):
    height = 128
    width = 128

    def __init_transformer(self):
        transform_ftn = []

        # transform_ftn.append(A.Normalize())
        # RandomBrightness(p=0.2, limit=0.2),
        # RandomContrast(p=0.1, limit=0.2),

        if self._args['training']:
            transform_ftn.extend([
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7)
            ])

        if self._args['resize']:
            transform_ftn.append(
                A.PadIfNeeded(self.height, self.width, cv2.BORDER_REFLECT_101)
            )

        transform_ftn.append(ToTensorV2())
        transforms = A.Compose(transform_ftn)
        return transforms

    def __init__(self, csv_data, test_mode=False):
        self.csv_data = Path(csv_data)
        self.base_path = self.csv_data.

        self.data = pd.read_csv(csv_data)
        self.depths = [int(x) for x in self.data['depths']]
        self.imgs = [str(self.base_path / x) for x in self.data[img_col]]
        self.n_samples = len(self.imgs) if test_model else 100

        # check label info
        self.have_label = True if 'mask' in self.data else False
        if have_label:
            self.masks = [str(self.base_path / x) for x in self.data[label_col]]
            self.labels = [int(x) for x in self.data['labels']]
            self.have_label = True

        self.transforms = self.__init_transformer()
        self.idx = list(range(self.n_samples))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, id):
        index = self.idx[id]

        # depth normalize
        depth = (self.depths[index] - 506) / 209

        gray = np.array(Image.open(self.imgs[index]).convert('RGB')) / 255
        data = {'image': img}

        if self.have_label:
            mask = Image.open(self.masks[index]).convert('1')
            mask = np.array(mask).astype(int)
            data['mask'] = mask

            label = self.labels[index]

            data = self.transforms(**data)

            return (data['image'].float(), depth), (data['mask'].squeeze(0).long(), label)

        data = self.transforms(**data)
        return (data['image'].float(), depth)
