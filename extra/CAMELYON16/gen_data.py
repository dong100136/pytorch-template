import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import random

from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('output_path', default=None, metavar='output_PATH', type=str,
                    help='output path')
parser.add_argument('--random', default=True, metavar='random', type=str2bool,
                    help='random choice point in the wsi')


class WSI:
    def __init__(self, wsi_path, output_path, level=6):
        self.wsi_name = wsi_path.stem
        self.wsi_path = wsi_path

        self.output_path = Path(output_path)
        self.tissue_mask_path = self.output_path / "tissue_mask" / (self.wsi_name + '.npy')
        self.tumor_mask_path = self.output_path / "tumor_mask" / (self.wsi_name + '.npy')
        self.non_tumor_mask_path = self.output_path / "non_tumor_mask" / (self.wsi_name + '.npy')

        self.tumor_data_path = self.output_path / "tumor"
        self.normal_data_path = self.output_path / "normal"
        self.tumor_data_list = self.output_path / "tumor_all_data.list"
        self.normal_data_list = self.output_path / "normal_all_data.list"

        self.tissue_mask_path.parent.mkdir(parents=True, exist_ok=True)
        self.tumor_mask_path.parent.mkdir(parents=True, exist_ok=True)
        self.non_tumor_mask_path.parent.mkdir(parents=True, exist_ok=True)
        self.tumor_data_path.mkdir(parents=True, exist_ok=True)
        self.normal_data_path.mkdir(parents=True, exist_ok=True)
        self.tumor_data_list.touch()
        self.normal_data_list.touch()

        self.wsi = openslide.OpenSlide(str(self.wsi_path))
        self.level = level
        self.RGB_min = 50

    def get_patch_point(self, mask_path, save_path, number=1000, random=True):
        mask_tissue = np.load(mask_path)
        X_idcs, Y_idcs = np.where(mask_tissue)

        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)

        if centre_points.shape[0] > number and random:
            sampled_points = centre_points[np.random.randint(centre_points.shape[0],
                                                             size=number), :]
        else:
            print("not using random choice...")
            sampled_points = centre_points

        sampled_points = (sampled_points * 2 ** self.level).astype(np.int32)  # make sure the factor

        name = np.full((sampled_points.shape[0], 1), self.wsi_path)
        center_points = np.hstack((name, sampled_points))

        with open(save_path, "a") as f:
            np.savetxt(f, center_points, fmt="%s", delimiter=",")

    def gen_non_tumor_mask(self):
        tissue_mask = np.load(self.tissue_mask_path)
        if self.tumor_mask_path.exists():
            tumor_mask = np.load(self.tumor_mask_path)
            normal_mask = tissue_mask & (~ tumor_mask)
        else:
            normal_mask = tissue_mask

        np.save(self.non_tumor_mask_path, normal_mask)

    def gen_tissue_mask(self):
        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = np.transpose(np.array(self.wsi.read_region((0, 0),
                                                             self.level,
                                                             self.wsi.level_dimensions[self.level]).convert('RGB')),
                               axes=[1, 0, 2])

        img_HSV = rgb2hsv(img_RGB)

        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > self.RGB_min
        min_G = img_RGB[:, :, 1] > self.RGB_min
        min_B = img_RGB[:, :, 2] > self.RGB_min

        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

        np.save(self.tissue_mask_path, tissue_mask)

    def gen_tumor_mask(self, json_path):
        # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
        w, h = self.wsi.level_dimensions[self.level]
        mask_tumor = np.zeros((h, w))  # the init mask, and all the value is 0

        # get the factor of level * e.g. level 6 is 2^6
        factor = self.wsi.level_downsamples[self.level]

        with open(json_path) as f:
            dicts = json.load(f)

        tumor_polygons = dicts['positive']

        for tumor_polygon in tumor_polygons:
            # plot a polygon
            name = tumor_polygon["name"]
            vertices = np.array(tumor_polygon["vertices"]) / factor
            vertices = vertices.astype(np.int32)

            cv2.fillPoly(mask_tumor, [vertices], (255))

        mask_tumor = mask_tumor[:] > 127
        mask_tumor = np.transpose(mask_tumor)

        np.save(self.tumor_mask_path, mask_tumor)


def main():
    args = parser.parse_args()

    wsi_dir = Path(args.wsi_path)

    json_fullpath = list(wsi_dir.glob("**/*.json"))
    json_fullpath = {x.stem: x for x in json_fullpath}
    wsi_fullpath = list(wsi_dir.glob("**/*.tif"))
    random.shuffle(wsi_fullpath)

    train_data_path = Path(args.output_path)
    train_data_path.mkdir(parents=True, exist_ok=True)
    for wsi_path in tqdm(wsi_fullpath):
        wsi_name = wsi_path.stem

        wsi = WSI(wsi_path, train_data_path)
        wsi.gen_tissue_mask()
        if wsi_name in json_fullpath:
            wsi.gen_tumor_mask(json_fullpath[wsi_name])
            wsi.get_patch_point(wsi.tumor_mask_path, wsi.tumor_data_list, 1000, random=args.random)

        wsi.gen_non_tumor_mask()
        wsi.get_patch_point(wsi.non_tumor_mask_path, wsi.normal_data_list, 400, random=args.random)


if __name__ == "__main__":
    main()
