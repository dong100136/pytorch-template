import pytest
from module import *
import logging

logging.basicConfig(level=logging.DEBUG)


# def test_kaggle_digit():
#     log = logging.getLogger("DigitDataLoader")
#     dataloader = DATA_LOADER['DigitDataLoader'](
#         csv_path='/root/dataset/digit-recognizer/train.csv',
#         batch_size=2
#     )

#     log.error(iter(dataloader).next())
#     for data, label in dataloader:
#         # log.error(data.shape)
#         # log.error(label.shape)

#         break

#     assert True


def test_kaggle_dog_vs_cat():
    log = logging.getLogger("")
    dataloader = DATA_LOADER['ImageFolderLoader'](
        train_data_dir="/root/dataset/dogs-vs-cats-redux-kernels-edition/train",
        batch_size=64,
        training=True,
        num_workers=4,
        test_mode=True,
        split=0.2
    )

    log.error(iter(dataloader).next())
    for data, label in dataloader:
        # log.error(data.shape)
        # log.error(label.shape)

        break

    assert False


if __name__ == '__main__':
    pytest.main()
