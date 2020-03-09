import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import PIL
from albumentations import (
    Resize, Normalize,
    Compose, HorizontalFlip, RandomBrightness,
    RandomContrast, ShiftScaleRotate)
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from ...registry import DATA_LOADER
from ..BaseDataLoader import BaseDataLoader


@DATA_LOADER.register("CarvanaDataLoader")
class CarvanaDataLoader(BaseDataLoader):
    test_mode = False

    def __init__(self, train_csv, valid_csv=None, *args, **kwargs):
        self.train_dataset = CSVImgDataSet(
            train_csv, data_aug=kwargs['training'],
            test_mode=self.test_mode,
            **kwargs)
        if valid_csv == None or self.test_mode == True:
            self.valid_dataset = self.train_dataset
        else:
            self.valid_dataset = CSVImgDataSet(valid_csv, **kwargs)

        return super().__init__(self.train_dataset, self.valid_dataset, *args, **kwargs)


class CSVImgDataSet(Dataset):
    def __init_transformer(self, **kwargs):
        transform_ftn = [
            Resize(self.img_size[0], self.img_size[1]),
            Normalize(
                mean=[0.6972476, 0.68982047, 0.68237394],
                std=[0.24286218, 0.24695995, 0.2439977]
            ),
        ]

        # transform_ftn.append(A.Normalize())

        if self.data_aug:
            transform_ftn.extend([
                HorizontalFlip(p=0.5),
                RandomBrightness(p=0.2, limit=0.2),
                RandomContrast(p=0.1, limit=0.2),
                ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7)
            ])

        transform_ftn.append(ToTensor())
        transforms = Compose(transform_ftn)
        return transforms

    def __init__(self, csv_data, img_size,
                 data_aug=False,
                 test_mode=False, TTA=None,
                 **kwarg):
        self.csv_data = Path(csv_data)
        self.base_path = self.csv_data.parent
        self.data_aug = data_aug
        self.tta = TTA
        self.img_size = img_size

        self.data = pd.read_csv(csv_data)
        self.imgs = [str(self.base_path / x) for x in self.data['images']]
        self.datalist = [x.stem for x in self.data['images']]
        self.n_samples = len(self.imgs) if not test_mode else 100

        # check label info
        self.have_label = True if 'masks' in self.data else False
        if self.have_label:
            self.masks = [str(self.base_path / x) for x in self.data['masks']]

        self.transforms = self.__init_transformer()
        self.idx = list(range(self.n_samples))

    def __len__(self):
        return self.n_samples

    def __tta(self, img):
        """[summary]

        Arguments:
            img {[PIL.Image]} -- input image

        Returns:
            img -- output image after tta
        """
        if self.tta == None:
            return img
        elif self.tta == "FLIP_UD":
            img = PIL.ImageOps.mirror(img)
            return img
        else:
            raise Exception("unknown tta type [%s]" % self.tta)

    def __getitem__(self, id):
        index = self.idx[id]

        gray = Image.open(self.imgs[index]).convert('RGB')
        gray = self.__tta(gray)
        gray = np.array(gray)

        data = {'image': gray}

        if self.have_label:
            mask = Image.open(self.masks[index]).convert('1')
            # mask = self.__tta(mask)
            mask = np.array(mask).astype(int)
            data['mask'] = mask

            data = self.transforms(**data)

            return data['image'].float(), data['mask'].squeeze(0).long()

        data = self.transforms(**data)
        return data['image'].float()


if __name__ == "__main__":
    train_csv = "/root/dataset/carvana-image-masking-challenge/workspace/train.csv.fold0"
    params = {
        'train_csv': train_csv,
        'valid_csv': train_csv,
        'training': False,
        'img_size': [320, 480]
    }
    dataloader = CarvanaDataLoader(**params)
    for img, mask in dataloader:
        print(img.mean(), img.shape)
        print(mask.float().mean(), mask.shape)

        print(img[0, :, :, :])

        break
