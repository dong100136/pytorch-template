import argparse
import pathlib
import random
import numpy as np
import cv2
from tqdm import tqdm


parser = argparse.ArgumentParser(description='calc mean and std from given paths')
parser.add_argument('paths', default=None, help="Path to the png")
parser.add_argument('sample_num', default=100, type=int)


if __name__ == "__main__":
    args = parser.parse_args()

    path = pathlib.Path(args.paths)
    image_list = list(path.glob("**/*.jpg"))
    print("find {} samples, going to sample {}".format(len(image_list), args.sample_num))

    random.shuffle(image_list)

    image_list = image_list[:args.sample_num]

    imgs = [np.expand_dims(cv2.imread(str(x)), -1) for x in image_list]
    imgs = np.concatenate(imgs, -1)
    imgs = imgs.astype(np.float32) / 255.

    means, stdevs = [], []
    for i in tqdm(range(3)):
        pixels = imgs[:, :, i, :].ravel()  # flatten
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # cv2 : BGR
    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
