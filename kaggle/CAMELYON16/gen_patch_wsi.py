import sys
import os
import argparse
import logging
import time
from shutil import copyfile
import numpy as np
from multiprocessing import Pool, Value, Lock
from tqdm import tqdm
from functools import partial

import openslide

count = Value('i', 0)
lock = Lock()

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('wsi', default=None, metavar='wsi_path',
                    type=str, help='Path to the input list of coordinates')
parser.add_argument('save_path', default=None, metavar='save_path', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=224, type=int, help='patch size, '
                    'default 256')
# parser.add_argument('--stride', default=32, type=int, help='patch size, '
#                     'default 256')
parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=30, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(point, wsi_path, save_path,  patch_size=224, stride=36, level=0,nsamples=0):
    x,y = point
    slide = openslide.OpenSlide(wsi_path)
    img = slide.read_region(
        (x, y), level,
        (patch_size, patch_size)
    ).convert('RGB')
    img.save(os.path.join(save_path, str(x) + '_' + str(y) + '.jpg'))

    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {}/{} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value,
                                 nsamples))


def run(args):
    logging.basicConfig(level=logging.INFO)

    slide = openslide.OpenSlide(args.wsi)
    width, height = slide.level_dimensions[args.level]
    args.stride = (args.patch_size-224+32)
    nsamples = len(np.arange(0, width, args.stride)) * len(np.arange(0, height, args.stride))

    img_list = []
    for x in np.arange(0, width, args.stride):
        for y in np.arange(0, height, args.stride):
            img_list.append((x, y))

    pool = Pool(processes=args.num_process)
    pool.map(
        partial(process,
                wsi_path=args.wsi,
                save_path=args.save_path,
                patch_size=args.patch_size,
                stride=args.stride,
                level = args.level,
                nsamples= nsamples),
        img_list
    )


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
