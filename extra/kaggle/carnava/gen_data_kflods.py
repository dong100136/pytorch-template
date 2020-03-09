import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

basepath = Path("/root/dataset/carvana-image-masking-challenge")
train_csv = basepath / 'workspace' / 'train.csv'
valid_csv = basepath / 'workspace' / 'valid.csv'
test_csv = basepath / 'workspace' / 'test.csv'
workspace_path = train_csv.parent

train_csv.parent.mkdir(exist_ok=True, parents=True)


imgs = list((basepath / "train").glob("*.jpg"))
# imgs = {x.stem: str(x.relative_to(basepath)) for x in imgs}
# imgs = list(imgs.values())
random.shuffle(imgs)
mask_basepath = basepath / 'train_masks'

print("found %d imgs" % len(imgs))


def get_data(img_path):
    img_name = Path(img_path).stem
    # mask_path = mask_basepath / ('%s_mask.gif' % img_name)

    img_path = "../train/%s.jpg" % img_name
    mask_path = "../train_masks/%s_mask.gif" % img_name
    return img_path, mask_path


data = list(map(get_data, imgs))

kf = KFold(n_splits=5, shuffle=True)
for k, (train_indics, valid_indics) in enumerate(kf.split(data)):
    with open(str(train_csv) + '.fold%d' % k, 'w') as f:
        f.write("images,masks\n")
        for idx in train_indics:
            f.write("%s,%s\n" % (data[idx][0], data[idx][1]))

    with open(str(valid_csv) + '.fold%d' % k, 'w') as f:
        f.write("images,masks\n")
        for i in valid_indics:
            f.write("%s,%s\n" % (data[idx][0], data[idx][1]))

    print("cut %d train_data, %d valid_data in fold.%d" % (len(train_indics), len(valid_indics), k))
