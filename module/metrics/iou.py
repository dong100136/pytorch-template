import torch
import numpy as np
from .utils import make_one_hot
from ..registry import METRICS


@METRICS.register("IOU")
def IOU(predicts, targets, threshold=0.5, eps=1e-5):
    """[summary]
    calc binary classify iou

    Arguments:
        predict {[torch.Tensor]} -- [N,H,W]
        target {[torch.Tensor]} -- [N,H,W]
    """
    assert predicts.shape == targets.shape, "the shape of prediction and targets shoudl be the same size"

    with torch.no_grad():
        predicts[predicts > threshold] = 1
        predicts[predicts <= threshold] = 0
        predicts = predicts.long()
        intersection = (predicts & targets).sum().float()
        union = (predicts | targets).sum().float()

        return intersection / (union + eps)


@METRICS.register("mIOU")
def mIOU(predicts, targets, threshold=0.5, n_classes=2):
    """mean IOU

    Arguments:
        predicts {[torch.Tensor]} -- [N,C,H,W]
        targets {[torch.Tensor]} -- [N,H,W]
        threshold {[float]} -- 
    """
    targets = make_one_hot(targets, n_classes)

    with torch.no_grad():
        mean_iou = 0
        for i in range(n_classes):
            p = predicts[:, i, :, :]
            t = targets[:, i, :, :]
            iou = IOU(p, t, threshold)
            mean_iou += iou / n_classes

    return mean_iou


@METRICS.register("kaggle_IOU")
def kaggle_iou(predicts, targets):
    scores = []
    for threshold in np.arange(0.5, 1, 0.05):
        scores.append(float(IOU(predicts, targets, threshold=threshold)))
    return np.mean(scores)
