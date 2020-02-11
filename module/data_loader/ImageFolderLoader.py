import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from ..registry import DATA_LOADER
from collections import namedtuple


@DATA_LOADER.register("ImageFolderLoader")
class ImageFolderLoader(DataLoader):
    """
    Base class for all data loaders
        config:
        [
            type: ImageFolderLoader
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
        "valid_data_dir": None,
        "batch_size": 16,
        "num_workers": 1,
        "imgs_mean": (0.485, 0.456, 0.406),  # imagenet
        "imgs_std": (0.229, 0.224, 0.225),  # imagenet
        "training": True,
        "split": 0,
        "test_mode": False,
        "collate_fn": default_collate,
        "resize": None
    }

    def __init__(self, train_data_dir, training, *args, **kwargs):
        self._args.update(kwargs)
        self._args['training'] = training

        self.batch_idx = 0
        transform_ftn = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=self._args['imgs_mean'],
                std=self._args['imgs_std']
            )
        ]
        if self._args['resize']:
            transform_ftn.insert(0,torchvision.transforms.Resize(self._args['resize']))

        self.tarnsforms = torchvision.transforms.Compose(transform_ftn)

        self.dataset = torchvision.datasets.ImageFolder(
            root=train_data_dir,
            transform=self.tarnsforms)
        self.n_samples = len(self.dataset)

        print("get %d data for train_data" % (self.n_samples))

        self.train_sampler, self.valid_sampler = self.__gen_sampler(self.dataset)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self._args['batch_size'],
            'sampler': self.train_sampler,
            'collate_fn': self._args['collate_fn'],
            'num_workers': self._args['num_workers'],
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)

    def __gen_sampler(self, dataset):
        idx_full = np.arange(self.n_samples)

        if self._args['training']:
            self.logger.info("using shuffled dataset")
            np.random.seed(0)
            np.random.shuffle(idx_full)

        if self._args['test_mode']:
            print("using test mode in dataloader")
            idx_full = idx_full[:100 * self._args['batch_size']]

        if isinstance(self._args['valid_data_dir'], str):
            len_valid = 0
        elif isinstance(self._args['split'], int):
            assert self._args['split'] < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = self._args['split']
        else:
            len_valid = int(len(idx_full) * self._args['split'])

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        print(train_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self._args['valid_data_dir']:
            data_dir = self._args['valid_data_dir']
            dataset = torchvision.datasets.ImageFolder(
                root=data_dir,
                transform=self.tarnsforms)
            valid_sampler, _ = self.__gen_sampler(dataset)
            print("get %d data for valid_data" % (len(dataset)))
        else:
            dataset = self.dataset
            valid_sampler = self.valid_sampler

        init_kwargs = {
            'dataset': dataset,
            'batch_size': self._args['batch_size'],
            'shuffle': False,
            'sampler': valid_sampler,
            'collate_fn': self._args['collate_fn'],
            'num_workers': self._args['num_workers'],
            'pin_memory': True
        }
        return DataLoader(**init_kwargs)
