import pytest
from data_loader import CIFAR100DataLoader
import logging

logging.basicConfig(level=logging.DEBUG)

def test_cifar10():
    log = logging.getLogger("test_cifar10")
    dataloader = CIFAR100DataLoader("/tmp/data",10)
    for data,label in dataloader:
        log.error(data.shape)
        log.error(label.shape)

        break

    assert False


if __name__ == '__main__':
    pytest.main()