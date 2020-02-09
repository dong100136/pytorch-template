import pytest
from module import *
import logging

logging.basicConfig(level=logging.DEBUG)

def test_kaggle_digit():
    log = logging.getLogger("DigitDataLoader")
    dataloader = DATA_LOADER['DigitDataLoader'](
        csv_path='/root/dataset/digit-recognizer/train.csv',
        batch_size=2
    )

    log.error(iter(dataloader).next())
    for data,label in dataloader:
        log.error(data.shape)
        log.error(label.shape)

        break

    assert False


if __name__ == '__main__':
    pytest.main()