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
    test_mode = False

    def __init__(self, train_csv, valid_csv=None, *args, **kwargs):
        self.train_dataset = CSVImgDataSet(train_csv, data_aug=kwargs['training'], test_mode=self.test_mode)
        if valid_csv == None or self.test_mode == True:
            self.valid_dataset = self.train_dataset
        else:
            self.valid_dataset = CSVImgDataSet(valid_csv)

        return super().__init__(self.train_dataset, self.valid_dataset, *args, **kwargs)


class CSVImgDataSet(Dataset):
    def __init_transformer(self):
        transform_ftn = []

        # transform_ftn.append(A.Normalize())
        # RandomBrightness(p=0.2, limit=0.2),
        # RandomContrast(p=0.1, limit=0.2),

        if self.data_aug:
            transform_ftn.extend([
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7)
            ])
        transform_ftn.append(
            A.PadIfNeeded(128, 128, cv2.BORDER_REFLECT_101)
        )

        transform_ftn.append(ToTensorV2())
        transforms = A.Compose(transform_ftn)
        return transforms

    def __init__(self, csv_data, data_aug=False, test_mode=False):
        self.csv_data = Path(csv_data)
        self.base_path = self.csv_data.parent
        self.data_aug = data_aug

        self.data = pd.read_csv(csv_data)
        self.depths = [int(x) for x in self.data['depths']]
        self.imgs = [str(self.base_path / x) for x in self.data['images']]
        self.n_samples = len(self.imgs) if not test_mode else 100

        # check label info
        self.have_label = True if 'masks' in self.data else False
        if self.have_label:
            self.masks = [str(self.base_path / x) for x in self.data['masks']]
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
        data = {'image': gray}

        if self.have_label:
            mask = Image.open(self.masks[index]).convert('1')
            mask = np.array(mask).astype(int)
            data['mask'] = mask

            label = self.labels[index]

            data = self.transforms(**data)

            return (data['image'].float(), depth), (data['mask'].squeeze(0).long(), label)

        data = self.transforms(**data)
        return (data['image'].float(), depth)


if __name__ == "__main__":
    train_csv = "/root/dataset/tgs-salt-identification-challenge/train.csv"
    dataloader = TgsDataLoader(train_csv, train_csv)
    for (img, depth), (mask, label) in dataloader:
        print(img.mean(), img.shape)
        print(depth.shape)
        print(mask.float().mean(), mask.shape)
        print(label.shape)

        break
