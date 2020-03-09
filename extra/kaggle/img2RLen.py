from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append("/root/project/paper/pytorch-template")
from utils.img_tools import RLenc
from tqdm import tqdm

# -------------------------------------------
img_id_list = list(pd.read_csv("/root/dataset/tgs-salt-identification-challenge/test.csv")['images'].values)
img_id_list = [Path(x).stem for x in img_id_list]
base_path = Path("/root/trash/log/prediction/SaltNet_V11_merge")
# tta = ['mask', 'mask_flip_ud']
tta = ['mask']
img_names = [x.stem for x in (base_path / 'mask').glob("*.npy")]
output_file = base_path / 'submission.csv'
threshold = 0.45
# -------------------------------------------

id = []
rle_mask = []

print(base_path)
print(tta)
print("threshold=", threshold)


def get_img_by_threshold(img, tta=None, threshold=0.5):
    if tta.find('flip_ud') >= 0:
        img = cv2.flip(img, 1)

    mask = np.zeros_like(img, dtype=np.float)
    mask[img <= threshold] = 0
    mask[img > threshold] = 1

    mask = mask[13:114, 13:114]
    return mask


for img_name in tqdm(img_names):
    if img_name not in img_id_list:
        continue

    masks = []
    for t in tta:
        img_path = base_path / t / ("%s.npy" % img_name)
        img = np.load(img_path)
        img = get_img_by_threshold(img, tta=t, threshold=threshold)
        masks.append(img)
    masks = np.array(masks)
    predict_mask = np.mean(masks, axis=(0))

    rLen = RLenc(predict_mask)
    id.append(img_name)
    rle_mask.append(rLen)
    del masks, predict_mask

data = pd.DataFrame({
    'id': id,
    'rle_mask': rle_mask
})
data.to_csv(output_file, index=None)

print("save %d lines" % (len(data)))
