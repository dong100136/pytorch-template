import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.plot_utils import make_grid_img

from ..registry import HOOK
from .base_hook import BaseHook
from abc import overide


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


@HOOK.register("SaveTensor")
class SaveTensor(BaseHook):
    #! not test
    def __init__(self, workspace):
        self.wordspace = workspace
        self.idx = 0
        self._predicts = None
        self._targets = None

    @overide
    def __after_epoch_hook(self, predicts, targets=None, name=None, **kwargs)
       with torch.no_grad():
            predicts = predicts.numpy()
            targets = targets.numpy()

        if not isinstance(self._predicts, np.array):
            self._predicts = predicts
            self._targets = targets
        else:
            self._predicts = np.concatenate([self._predicts,predicts],axis=0)
            self._targets = np.concatenate([self._targets,targets],axis=0)

    @overide
    def __after_epoch_hook(self):
        np.save(self.workspace / "predicts.npy", self._predicts)
        np.save(self.workspace / "targets.npy", self._targets)

@HOOK.register("SaveSegTensor")
class SaveSegTensor(BaseHook):
    def __init__(self, workspace,predict_path="masks",target_workspace="targets", *kwargs):
        """[prediction hooks for segmentation task]
        Keyword Arguments:
            predict_path {str} -- [the path to save prediction masks] (default: {"masks"})
            target_workspace {str} -- [the path to save target masks] (default: {"targets"})
        """
        self.predict_path = workspace_path /predict_path
        self.target_path = workspace / target_workspace
        self.idx = 0

    @overide
    def __after_epoch_hook(self, predicts, targets=None, imgs=None, **kwargs):
        with torch.no_grad():
            predicts = predicts.numpy()
            targets = targets.numpy() if targets else None

        n_samples = predicts.shape[0]
        for i in range(n_samples):
            if imgs:
                img_name = imgs[self.idx]
            else:
                img_name = self.idx

            self.idx += 1
            
            np.save(self.predict_path/("%s.npy"%img_name),predicts[i])

            if targets:
                np.save(self.target_path/("%s.npy"%img_name),targets[i])


