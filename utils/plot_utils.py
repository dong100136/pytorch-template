import matplotlib.pyplot as plt
import random


def make_grid_img(data, labels, nrows=3, ncols=3, shuffle=True, save_path=None):
    '''
    input:
        data: [N,H,W,C]
        label: [N]
    '''
    num = data.shape[0]
    nsample = nrows * ncols
    if shuffle:
        idx = [random.randint(0, num - 1) for _ in range(nsample)]
    else:
        idx = [x for x in range(nsample)]

    plt.figure(figsize=(nrows * 2, ncols * 2), dpi=200)
    for i, id in enumerate(idx):
        plt.subplot(nrows, ncols, i + 1)
        img = data[id]
        label = labels[id]

        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.title(label)

    if save_path:
        plt.savefig(save_path,bbox_inches = 'tight')
