import matplotlib.pyplot as plt
import numpy as np
import random
import cv2


def make_grid_img(imgs_path, labels, prob=None, nrows=3, ncols=3, shuffle=True, save_path=None):
    '''
    input:
        data: [N,H,W,C]
        label: [N]
    '''
    num = len(labels)
    nsample = nrows * ncols
    if shuffle:
        idx = [random.randint(0, num - 1) for _ in range(nsample)]
    else:
        idx = [x for x in range(nsample)]

    plt.figure(figsize=(nrows * 2, ncols * 2), dpi=200)
    for i, id in enumerate(idx):
        plt.subplot(nrows, ncols, i + 1)

        img = cv2.imread(imgs_path[id])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = labels[id]

        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        title = label
        if isinstance(prob, np.ndarray):
            title+='(%f)'%prob[id]
        plt.title(title)

    if save_path:
        plt.savefig(save_path,bbox_inches = 'tight')
