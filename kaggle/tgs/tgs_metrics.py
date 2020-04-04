import torch
import numpy as np
from ...registry import METRICS
from ..accuracy import accuracy
from ..iou import kaggle_iou


@METRICS.register("tgs_accuracy")
def tgs_accuracy(predicts, targets, threshold=0.5):
    masks = torch.sigmoid(predicts[0])
    return accuracy(masks, targets[0], threshold)


@METRICS.register("tgs_kaggle_iou")
def tgs_kaggle_iou(predicts, targets):
    masks = torch.sigmoid(predicts[0])
    return kaggle_iou(masks, targets[0])


@METRICS.register("tgs_kaggle_iou_mask")
def tgs_kaggle_iou_mask(predicts, targets):
    with torch.no_grad():
        masks = predicts[0]
        masks = torch.sigmoid(masks)
        labels = predicts[1]
        for i in range(len(masks)):
            if labels[i] <= 0.5:
                masks[i] = 0

        return kaggle_iou(masks, targets[0])


@METRICS.register("tgs_accurary_classify")
def tgs_accuracy_classify(predicts, targets, threshold=0.5):
    return accuracy(predicts[1], targets[1], threshold)
