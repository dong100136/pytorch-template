import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from ..registry import DATA_LOADER

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

    def __init__(self, train_data_dir, 
                 batch_size, num_workers,
                 imgs_mean, imgs_std, shuffle,
                 valid_data_dir = None,
                 split=0.2,
                 size=(224,224),
                 test_mode=False,
                 collate_fn=default_collate):
        self.batch_size = batch_size
        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.test_mode = test_mode
        self.shuffle = shuffle
        self.size = size
        self.split = split

        self.batch_idx = 0

        self.imgs_mean = imgs_mean
        self.imgs_std = imgs_std

        dataset = torchvision.datasets.ImageFolder(root=train_data_dir,
                                                   transform=torchvision.transforms.Compose([
                                                    #    torchvision.transforms.Resize(size, interpolation=2),
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(
                                                           mean=imgs_mean,
                                                           std=imgs_std
                                                       )
                                                   ]))

        print("get %d data for train_data" % (len(dataset)))

        train_sampler,valid_sampler = self.__gen_sampler(dataset, test_mode)
        self.valid_sampler = valid_sampler

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'sampler': train_sampler,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)

    def __gen_sampler(self, dataset, test_mode=False):
        n_samples = len(dataset)
        self.n_samples = len(dataset)

        idx_full = np.arange(n_samples)

        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(idx_full)

        if test_mode:
            print("using test mode in dataloader")
            idx_full = idx_full[:self.batch_size]
        else:
            if isinstance(self.split, int):
                assert self.split > 0
                assert self.split < self.n_samples, "validation set size is configured to be larger than entire dataset."
                len_valid = self.split
            else:
                len_valid = int(self.n_samples * self.split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.shuffle = False
        # return SubsetRandomSampler(idx_full)
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_data_dir is None:
            return None
        else:
            data_dir = self.valid_data_dir if not self.test_mode else self.train_data_dir
            dataset = torchvision.datasets.ImageFolder(root=data_dir,
                                                       transform=torchvision.transforms.Compose([
                                                        #    torchvision.transforms.Resize(self.size, interpolation=2),
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               mean=self.imgs_mean,
                                                               std=self.imgs_std
                                                           )
                                                       ]))
            print("get %d data for valid_data" % (len(dataset)))
            # sampler = self.__gen_sampler(dataset, self.test_mode)
            init_kwargs = {
                'dataset': dataset,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'sampler': self.valid_sampler,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                'pin_memory': True
            }
            return DataLoader(**init_kwargs)
