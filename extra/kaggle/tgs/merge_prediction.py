from pathlib import Path
import numpy as np
from tqdm import tqdm

output_name = "mask"
basepath = Path("/root/trash/log/prediction")
output = basepath / 'SaltNet_V11_merge' / output_name
folds_dir = [basepath / ('SaltNet_V11_fold%d' % i)for i in range(5)]
img_names = [img.stem for img in folds_dir[0].glob("%s/*.npy" % output_name)]

output.mkdir(parents=True, exist_ok=True)

print(output)

print("fint %d images" % len(img_names))
for img in tqdm(img_names):
    imgs = [np.load(p / output_name / ('%s.npy' % img)) for p in folds_dir]
    imgs = np.array(imgs)
    merge_imgs = np.mean(imgs, axis=(0))
    np.save(output / ('%s.npy' % img), merge_imgs)
