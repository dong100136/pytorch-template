from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append("/root/project/paper/pytorch-template")
from utils.img_tools import mask2rle
from tqdm import tqdm

# -------------------------------------------
img_id_list = list(pd.read_csv("/root/dataset/carvana-image-masking-challenge/workspace/test.csv")['images'].values)
img_id_list = [Path(x).name for x in img_id_list]
base_path = Path("/root/trash/log/prediction/CarNet_v0")
predict_list = ['%d.npy' % i for i in range(len(img_id_list))]
# tta = ['mask', 'mask_flip_ud']
tta = ['masks']
output_file = base_path / 'submission.csv'
threshold = 0.45
# -------------------------------------------

rle_mask = []

print(base_path)
print(tta)
print("threshold=", threshold)


def get_img_by_threshold(img, tta=None, threshold=0.5):
    if tta.find('flip_ud') >= 0:
        img = cv2.flip(img, 1)

    mask = np.zeros_like(img, dtype=np.int8)
    mask[img <= threshold] = 0
    mask[img > threshold] = 1

    return mask


for img_name in tqdm(predict_list):
    masks = []
    for t in tta:
        img_path = base_path / t / img_name
        img = np.load(img_path)
        img = get_img_by_threshold(img, tta=t, threshold=threshold)
        masks.append(img)
    masks = np.array(masks)
    predict_mask = np.mean(masks, axis=(0))

    rLen = mask2rle(predict_mask)
    rle_mask.append(rLen)
    del masks, predict_mask

print(len(img_id_list), ',', len(rle_mask))
data = pd.DataFrame({
    'img': img_id_list,
    'rle_mask': rle_mask
})
data.to_csv(output_file, index=None)

print("save %d lines" % (len(data)))
