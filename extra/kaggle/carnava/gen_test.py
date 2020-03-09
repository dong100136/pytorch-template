import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

basepath = Path("/root/dataset/carvana-image-masking-challenge")
test_csv = basepath / 'workspace' / 'test.csv'

imgs = list((basepath / "test").glob("*.jpg"))
random.shuffle(imgs)

print("found %d imgs" % len(imgs))

with open(test_csv, 'w') as f:
    f.write("images\n")
    for img_path in imgs:
        img_name = img_path.stem
        img_path = "../test/%s.jpg" % img_name
        f.write("%s\n" % (img_path))
