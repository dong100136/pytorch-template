import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class ImageFolderLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, train_data_dir, valid_data_dir,
                 batch_size, num_workers,
                 imgs_mean, imgs_std, shuffle,
                 test_mode=False,
                 collate_fn=default_collate):
        self.batch_size = batch_size
        self.valid_data_dir = valid_data_dir
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.test_mode = test_mode
        self.shuffle = shuffle

        self.batch_idx = 0

        dataset = torchvision.datasets.ImageFolder(root=train_data_dir,
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(
                                                           mean=imgs_mean,
                                                           std=imgs_std
                                                       )
                                                   ]))

        print("get %d data for train_data" % (len(dataset)))

        sampler = self.__gen_sampler(dataset, test_mode)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'sampler': sampler,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)

    def __gen_sampler(self, dataset, test_mode=False):
        n_samples = len(dataset)

        idx_full = np.arange(n_samples)

        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(idx_full)

        if test_mode:
            print("using test mode in dataloader")
            idx_full = idx_full[:16]

        self.shuffle = False
        return SubsetRandomSampler(idx_full)

    def split_validation(self):
        if self.valid_data_dir is None:
            return None
        else:
            dataset = torchvision.datasets.ImageFolder(root=self.valid_data_dir,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               mean=[0.6600297, 0.4745953, 0.6561866],
                                                               std=[0.22620466, 0.2393456, 0.18473646]
                                                           )
                                                       ]))
            print("get %d data for valid_data" % (len(dataset)))
            sampler = self.__gen_sampler(dataset, self.test_mode)
            init_kwargs = {
                'dataset': dataset,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'sampler': sampler,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers,
                'pin_memory': True
            }
            return DataLoader(**init_kwargs)
