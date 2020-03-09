import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

basepath = Path("/root/dataset/tgs-salt-identification-challenge")
train_csv = basepath / 'train.csv'
valid_csv = basepath / 'valid.csv'
test_csv = basepath / 'test.csv'

bad_masks = ('1eaf42beee', '33887a0ae7', '33dfce3a76', '3975043a11', '39cd06da7d', '483b35d589', '49336bb17b', '4ef0559016', '4fbda008c7', '4fdc882e4b',
             '50d3073821', '53e17edd83', '5b217529e7', '5f98029612', '608567ed23', '62aad7556c', '62d30854d7', '6460ce2df7', '6bc4c91c27', '7845115d01',
             '7deaf30c4a', '80a458a2b6', '81fa3d59b8', '8367b54eac', '849881c690', '876e6423e6', '90720e8172', '916aff36ae', '919bc0e2ba', 'a266a2a9df',
             'a6625b8937', 'a9ee40cf0d', 'aeba5383e4', 'b63b23fdc9', 'baac3469ae', 'be7014887d', 'be90ab3e56', 'bfa7ee102e', 'bfbb9b9149', 'c387a012fc',
             'c98dfd50ba', 'caccd6708f', 'cb4f7abe67', 'd0bbe4fd97', 'd4d2ed6bd2', 'de7202d286', 'f0c401b64b', 'f19b7d20bb', 'f641699848', 'f75842e215',
             '00950d1627', '0280deb8ae', '06d21d76c4', '09152018c4', '09b9330300', '0b45bde756', '130229ec15', '15d76f1672', '182bfc6862', '23afbccfb5',
             '24522ec665', '285f4b2e82', '2bc179b78c', '2f746f8726', '3cb59a4fdc', '403cb8f4b3', '4f5df40ab2', '50b3aef4c4', '52667992f8', '52ac7bb4c1',
             '56f4bcc716', '58de316918', '640ceb328a', '71f7425387', '7c0b76979f', '7f0825a2f0', '834861f1b6', '87afd4b1ca', '88a5c49514', '9067effd34',
             '93a1541218', '95f6e2b2d1', '96216dae3b', '96523f824a', '99ee31b5bc', '9a4b15919d', '9b29ca561d', '9eb4a10b98', 'ad2fa649f7', 'b1be1fa682',
             'b24d3673e1', 'b35b1b412b', 'b525824dfc', 'b7b83447c4', 'b8a9602e21', 'ba1287cb48', 'be18a24c49', 'c27409a765', 'c2973c16f1', 'c83d9529bd',
             'cef03959d8', 'd4d34af4f7', 'd9a52dc263', 'dd6a04d456', 'ddcb457a07', 'e12cd094a6', 'e6e3e58c43', 'e73ed6e7f2', 'f6e87c1458', 'f7380099f6',
             'fb3392fee0', 'fb47e8e74e', 'febd1d2a67')

depths = pd.read_csv(basepath / 'depths.csv')

imgs = list((basepath / "train/images").glob("*.png"))
imgs = {x.stem: str(x.relative_to(basepath)) for x in imgs}

for x in bad_masks:
    imgs.pop(x)

imgs = list(imgs.values())
random.shuffle(imgs)


def filter_mask(img_path):
    img = Image.open(basepath / img_path).convert('1')
    img = np.array(img)

    n_pixels = img.size
    n_mask_pixels = np.count_nonzero(img)
    rate = float(n_mask_pixels) / n_pixels

    return 1 if n_mask_pixels > 5 else 0


def get_data(img_path):
    img_name = Path(img_path).stem
    depth = depths.loc[depths['id'] == img_name, 'z'].values[0]
    mask_path = img_path.replace('images', 'masks')
    label = filter_mask(img_path)
    return (img_path, depth, mask_path), label


data = list(map(get_data, imgs))
X = list(map(lambda x: x[0], data))
Y = list(map(lambda x: x[1], data))

kf = StratifiedKFold(n_splits=5)
for k, (train_indics, valid_indics) in enumerate(kf.split(X, Y)):
    with open(str(train_csv) + '.fold%d' % k, 'w') as f:
        f.write("images,depths,masks,labels\n")
        for i in train_indics:
            f.write("%s,%d,%s,%d\n" % (X[i][0], X[i][1], X[i][2], Y[i]))

    with open(str(valid_csv) + '.fold%d' % k, 'w') as f:
        f.write("images,depths,masks,labels\n")
        for i in valid_indics:
            f.write("%s,%d,%s,%d\n" % (X[i][0], X[i][1], X[i][2], Y[i]))

    print("cut %d train_data, %d valid_data in fold.%d" % (len(train_indics), len(valid_indics), k))