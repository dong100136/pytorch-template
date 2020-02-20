from ..registry import HOOK
from utils.plot_utils import make_grid_img
import numpy as np
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt


@HOOK.register("save_tensor")
def save_tensor(predicts, targets, samples=None,
                workspace=Path("/tmp"),
                *args, **kwargs):
    with torch.no_grad():
        predicts = predicts.numpy()
        targets = targets.numpy()

        np.save(workspace / "predicts.npy", predicts)
        np.save(workspace / "targets.npy", targets)

        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
            np.save(workspace / 'samples.npy', samples)
