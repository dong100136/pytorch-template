import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from utils.plot_utils import make_grid_img

from ..registry import HOOK
from .base_hook import BaseHook


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

    def __after_epoch_hook(self, predicts, targets=None, name=None, **kwargs):
        with torch.no_grad():
            predicts = predicts.numpy()
            targets = targets.numpy()

        if not isinstance(self._predicts, np.array):
            self._predicts = predicts
            self._targets = targets
        else:
            self._predicts = np.concatenate([self._predicts, predicts], axis=0)
            self._targets = np.concatenate([self._targets, targets], axis=0)

    def __after_epoch_hook(self):
        np.save(self.workspace / "predicts.npy", self._predicts)
        np.save(self.workspace / "targets.npy", self._targets)


@HOOK.register("SaveSegTensor")
class SaveSegTensor(BaseHook):
    def __init__(self,
                 workspace,
                 datalist=None,
                 predict_path="masks", target_path="targets",
                 img_size=None,
                 **kwargs):
        """[prediction hooks for segmentation task]
        Keyword Arguments:
            predict_path {str} -- [the path to save prediction masks] (default: {"masks"})
            target_workspace {str} -- [the path to save target masks] (default: {"targets"})
        """
        super(SaveSegTensor, self).__init__()
        workspace = Path(workspace)
        self.predict_path = workspace / predict_path
        self.target_path = workspace / target_path

        self.predict_path.mkdir(parents=True, exist_ok=True)
        self.target_path.mkdir(parents=True, exist_ok=True)
        self.idx = 0

        self.resize = isinstance(img_size, list)
        self.img_size = img_size

        self.datalist = datalist

    def save_img(self, img_path, img):
        if self.resize:
            img = cv2.resize(img,
                             (self.img_size[1], self.img_size[0]),
                             interpolation=cv2.INTER_NEAREST)

        np.save(img_path, img)

    def after_epoch_hook(self, predicts, targets=None, **kwargs):
        with torch.no_grad():
            predicts = predicts.numpy()
            targets = targets.numpy() if targets else None

        n_samples = predicts.shape[0]
        for i in range(n_samples):
            if imgs:
                img_name = self.datalist[self.idx]
            else:
                img_name = self.idx

            self.idx += 1

            img_path = self.predict_path / ("%s.npy" % img_name)
            self.save_img(img_path, predicts[i])

            if targets:
                img_path = self.target_path / ("%s.npy" % img_name)
                self.save_img(img_path, targets[i])

    def after_predict_hook(self):
        print("save %d data to %s" % (self.idx - 1, self.predict_path))
