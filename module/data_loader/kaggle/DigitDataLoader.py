from torchvision import datasets, transforms
from base import BaseDataLoader
import pandas as pd
import numpy as np
import torch
from ...registry import DATA_LOADER

@DATA_LOADER.register("DigitDataLoader")
class DigitDataLoader(BaseDataLoader):
    """
    This is for competition(https://www.kaggle.com/c/digit-recognizer)
    """
    def __init__(self, csv_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])

        # load data
        train = pd.read_csv(csv_path, dtype = np.float32)

        # split data into features(pixels) and labels(numbers from 0 to 9)
        if training:
            targets_numpy = train.label.values
            targets = torch.from_numpy(targets_numpy).long()
        else:
            shuffle = False

        features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization
        features_numpy = features_numpy.reshape((-1,1,28,28))
        features = torch.from_numpy(features_numpy)

        if training:
            self.dataset = torch.utils.data.TensorDataset(features,targets)
        else:
            self.dataset = torch.utils.data.TensorDataset(features)
    
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
