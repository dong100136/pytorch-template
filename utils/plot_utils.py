import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import click
from pathlib import Path
import random
from PIL import Image


def make_grid_img(imgs_path, labels=None, prob=None, nrows=3, ncols=3, shuffle=True, save_path=None):
    '''
    input:
        numpy array
    '''
    num = len(imgs_path)
    nsample = nrows * ncols
    if shuffle:
        idx = [random.randint(0, num - 1) for _ in range(nsample)]
    else:
        idx = [x for x in range(nsample)]

    plt.figure(figsize=(nrows * 2, ncols * 2), dpi=200)
    for i, id in enumerate(idx):
        plt.subplot(nrows, ncols, i + 1)

        # img = cv2.imread(imgs_path[id])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.open(imgs_path[id])
        print(id, imgs_path[id], img.size)

        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        if labels:
            label = labels[id]
            title = label
            if isinstance(prob, np.ndarray):
                title += '(%f)' % prob[id]
            plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


@click.command()
@click.option('--path', help="the imgs dir path")
@click.option('--extention', default='jpg', help="the file type of img")
@click.option('--save', default='/tmp/test.jpg', help='saving path')
def show_imgs_in(path, save='/tmp/test.jpg', extention='jpg', nrows=3, ncols=3):
    print(path)
    path = list(Path(path).glob("*.jpg"))
    print("found %d imgs" % (len(path)))
    selected_paths = random.choices(path, k=ncols * nrows)
    print("selects %d files from %d files" % (len(selected_paths), len(path)))

    make_grid_img(selected_paths, ncols=ncols, nrows=nrows, save_path=save)


if __name__ == '__main__':
    show_imgs_in()
