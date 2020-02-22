from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append("/root/project/paper/pytorch-template")
from utils.img_tools import RLenc
from tqdm import tqdm

# -------------------------------------------
base_path = Path("/root/trash/log/prediction/tgs-net-aux-depth-v6.1")
imgs_path = list((base_path / 'mask').glob("*.npy"))
output_file = base_path / 'submission.csv'
threshold = 0.5
# -------------------------------------------

id = []
rle_mask = []

for img_path in tqdm(imgs_path):
    img = np.load(img_path)
    img = cv2.resize(img, (101, 101))

    img[img <= threshold] = 0
    img[img > threshold] = 1

    rLen = RLenc(img)
    img_name = img_path.stem
    id.append(img_name)
    rle_mask.append(rLen)
    del img

data = pd.DataFrame({
    'id': id,
    'rle_mask': rle_mask
})
data.to_csv(output_file, index=None)
