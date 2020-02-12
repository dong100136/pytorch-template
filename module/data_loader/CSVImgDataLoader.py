import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from ..registry import DATA_LOADER
from collections import namedtuple
from pathlib import Path
import pandas as pd
from PIL import Image
import torch


@DATA_LOADER.register("CSVImgDataLoader")
class CSVImgDataLoader(DataLoader):
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
        "imgs_mean": (0.485, 0.456, 0.406),  # imagenet
        "imgs_std": (0.229, 0.224, 0.225),  # imagenet
        "training": True,
        "split": 0,
        "test_mode": False,
        "collate_fn": default_collate,
        "resize": None,
        "masks_col": 'masks',
        'img_col': 'images',
        'data_augement': False,
    }

    def __init__(self, train_csv, training, *args, **kwargs):
        self._args.update(kwargs)
        self._args['train_csv'] = train_csv
        self._args['training'] = training
        transforms = self.__init_transformer(self._args['data_augement'])

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
            'shuffle': training,
            'collate_fn': self._args['collate_fn'],
            'num_workers': self._args['num_workers'],
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)

    def __init_transformer(self, data_augement):
        transform_ftn = []
        if self._args['resize']:
            transform_ftn.append(T.Resize(self._args['resize']))

        if data_augement:
            transform_ftn.append(T.RandomRotation(20))
            transform_ftn.append(T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5))

        transform_ftn.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=self._args['imgs_mean'],
                std=self._args['imgs_std']
            )
        ])
        transforms = torchvision.transforms.Compose(transform_ftn)
        return transforms

    def split_validation(self):
        data_csv = self._args['valid_csv']
        transforms = self.__init_transformer(False)
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
    def __init__(self, csv_data, label_col='masks', img_col='images',
                 transforms=None, test_mode=False):
        self.csv_data = Path(csv_data)
        self.base_path = self.csv_data.parent

        self.data = pd.read_csv(csv_data)
        if label_col in self.data:
            self.masks = list(self.data[label_col])
            self.have_label = True
        else:
            self.have_label = False

        self.imgs = [str(self.base_path / x) for x in self.data[img_col]]
        self.n_samples = len(self.imgs)
        self.transforms = transforms

        if test_mode:
            self.n_samples = min(300, self.n_samples)
            self.imgs = self.imgs[:self.n_samples]
            if self.have_label:
                self.masks = self.masks[:self.n_samples]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(img)

        if self.have_label:
            mask = self.masks[index]

            return img, mask
        else:
            return img
