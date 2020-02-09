from torchvision import datasets, transforms
from base import BaseDataLoader
from ..registry import DATA_LOADER

mean = {
'cifar10': (0.4914, 0.4822, 0.4465),
'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
'cifar10': (0.2023, 0.1994, 0.2010),
'cifar100': (0.2675, 0.2565, 0.2761),
}

@DATA_LOADER.register("CIFAR100DataLoader")
class CIFAR100DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar100'], std['cifar100'])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

@DATA_LOADER.register("CIFAR10DataLoader")
class CIFAR10DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar10'], std['cifar10'])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)