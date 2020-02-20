from ..registry import HOOK
import random
from utils.plot_utils import make_grid_img
import numpy as np
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt


@HOOK.register("plot_seg")
def plot_seg(samples=None, predicts=None, targets=None,
             n_samples=5,
             shuffle=True,
             workspace=Path("/tmp"),
             save_path="plot_seg.png",
             *args, **kwargs):
    '''
    func: plot_result
    args:
      save_path: plot.png
    '''
    with torch.no_grad():
        save_path = workspace / save_path
        n_samples = min(n_samples, len(targets))

        idxs = list(range(len(targets)))
        random.shuffle(idxs)

        if len(predicts.shape) == 4:
            prob, predict_labels = torch.max(predicts, axis=1)
        else:
            prob = predicts
            predict_labels = torch.zeros_like(predicts)
            predict_labels[predicts > 0.5] = 1

        if isinstance(samples, torch.Tensor) and len(samples.shape) == 4:
            samples = samples.permute(0, 2, 3, 1)

        for id, i in enumerate(idxs[:n_samples]):
            if isinstance(samples, torch.Tensor):
                s = samples[i]
                plt.subplot(n_samples, 3, 3 * id + 1)
                val_min = torch.min(s)
                val_max = torch.max(s)
                s = (s - val_min) / (val_max - val_min)
                plt.imshow(s)
                plt.axis('off')

            if i == 0:
                plt.title("origin input")

            plt.subplot(n_samples, 3, 3 * id + 2)
            plt.imshow(predict_labels[i].long())
            plt.axis('off')

            if i == 0:
                plt.title("predict labels")

            plt.subplot(n_samples, 3, 3 * id + 3)
            plt.imshow(targets[i])
            plt.axis('off')

            if i == 0:
                plt.title("ground true")

        plt.savefig(save_path, bbox_inches='tight')
        print("plot seg result and save to %s" % save_path)
