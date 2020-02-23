import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from ...registry import DATA_LOADER
from collections import namedtuple
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from albumentations import (HorizontalFlip, ShiftScaleRotate, RandomContrast, RandomBrightness, Compose)
import random
import cv2


@DATA_LOADER.register("TgsDataLoader")
class TgsDataLoader(DataLoader):
    """
    Base class for all data loaders
        config:
        [
            type: CSVDataLoader
            args:
                train_data_dir: /root/dataset/workspace/CAMELYON16_v2/train
                valid_data_dir: /root/dataset/workspace/CAMELYON16_v2/valid
                batch_size: 64
                shuffle: True
                num_workers: 4
                test_mode: false # this mode will use 16 data for validate the model
                imgs_mean: [0.6600297, 0.4745953, 0.6561866]
                imgs_std: [0.22620466, 0.2393456, 0.18473646]
        ]
    """
    _args = {
        "valid_csv": None,
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

    def __init__(self, train_csv, training, *args, **kwargs):
        self._args.update(kwargs)
        self._args['train_csv'] = train_csv
        self._args['training'] = training
        transforms = self.__init_transformer()

        self.batch_idx = 0

        self.dataset = CSVImgDataSet(train_csv,
                                     label_col=self._args['masks_col'],
                                     img_col=self._args['img_col'],
                                     transforms=transforms,
                                     test_mode=self._args['test_mode'])
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
            h = self._args['resize'][0]
            w = self._args['resize'][1]
            transform_ftn.append(A.PadIfNeeded(h, w, cv2.BORDER_REFLECT_101))

        transform_ftn.append(ToTensorV2())
        transforms = A.Compose(transform_ftn)
        return transforms

    def split_validation(self):
        if self._args['test_mode']:
            return self
        else:
            data_csv = self._args['valid_csv']
            transforms = self.__init_transformer()
            dataset = CSVImgDataSet(data_csv,
                                    label_col=self._args['masks_col'],
                                    img_col=self._args['img_col'],
                                    transforms=transforms,
                                    test_mode=self._args['test_mode'])
            print("get %d data for valid_data" % (len(dataset)))

            init_kwargs = {
                'dataset': dataset,
                'batch_size': self._args['batch_size'],
                'shuffle': self._args['training'],
                'collate_fn': self._args['collate_fn'],
                'num_workers': self._args['num_workers'],
                'pin_memory': True
            }
            return DataLoader(**init_kwargs)


class CSVImgDataSet(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, csv_data, transforms,
                 label_col='masks', img_col='images',
                 test_mode=False):
        self.csv_data = Path(csv_data)
        self.base_path = self.csv_data.parent

        self.data = pd.read_csv(csv_data)
        if label_col in self.data:
            self.masks = [str(self.base_path / x) for x in self.data[label_col]]
            self.labels = [int(x) for x in self.data['labels']]
            self.have_label = True
        else:
            self.have_label = False

        self.depths = [int(x) for x in self.data['depths']]
        self.imgs = [str(self.base_path / x) for x in self.data[img_col]]
        self.n_samples = len(self.imgs)
        self.transforms = transforms

        self.idx = list(range(self.n_samples))
        # random.shuffle(self.idx)

        if test_mode:
            self.n_samples = min(10, self.n_samples)

        self.depth_info = self.__gen_depth_info()

    def __gen_depth_info(self):
        depth_info = np.arange(1, 102, 1).reshape((101, 1))
        depth_info = np.tile(depth_info, (1, 101))
        depth_info = depth_info / 101
        return depth_info

    def __len__(self):
        return self.n_samples

    def __getitem__(self, id):
        index = self.idx[id]
        depth = (self.depths[index] - 506) / 209

        gray = np.array(Image.open(self.imgs[index]).convert('L')) / 255
        img = np.ones((101, 101, 3))
        img[:, :, 0] = gray
        img[:, :, 1] = self.depth_info
        img[:, :, 2] = gray * self.depth_info
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
