from ..registry import HOOK
from utils.plot_utils import make_grid_img
import numpy as np
import os
from pathlib import Path


@HOOK.register("plot_seg")
def plot_seg(samples, predicts, targets,
             n_samples=5,
             shuffle=True,
             workspace=Path("/tmp"),
             save_path="plot.png",
             *args, **kwargs):
    '''
    func: plot_result
    args:
      save_path: plot.png
    '''
    with torch.no_grad():
        save_path = workspace / save_path
        n_samples = min(n_samples, len(targets))

        samples = samples[:n_samples]
        predicts = predicts[:n_samples]
        targets = targets[:n_samples]

        prob = torch.max(predicts, axis=1)
        predicts = torch.argmax(predicts, axis=1)

        samples = samples.transpose([0, 2, 3, 1])
        predicts = predicts.transpose([0, 2, 3, 1])
        targets = targets.transpose([0, 2, 3, 1])

        print(prob.shape, predict.shape, targets.shape)

        for i in range(n_samples):
            plt.subplot(n_samples, 3, 3 * i + 1)
            plt.imshow(sampels[i])
