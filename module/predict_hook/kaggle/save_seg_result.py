from ...registry import HOOK
from ...metrics.iou import kaggle_iou
import random
from utils.plot_utils import make_grid_img
import numpy as np
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt


@HOOK.register("save_seg_result")
def save_seg_result(dataset=None, predicts=None, targets=None,
                    workspace=Path("/tmp"),
                    file_name="seg_result.csv",
                    *args, **kwargs):
    with torch.no_grad():
        img_paths = dataset.imgs
        predict_masks = predicts[0]
        target_masks = targets[0]
        n_samples = len(predict_masks)

        np.save("/tmp/test.npy", target_masks.numpy())

        with open(workspace / file_name, 'w') as f:
            f.write("img_path,miou,label\n")

            for i in range(n_samples):
                pred = predict_masks[i].unsqueeze(dim=0)
                target = target_masks[i].unsqueeze(dim=0)
                score = kaggle_iou(pred, target)
                f.write("%s,%f,%f\n" % (img_paths[i], score, predicts[1][i]))

        print("save seg result in %s" % (workspace / file_name))
