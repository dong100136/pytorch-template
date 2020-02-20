import torch
import numpy as np
from ...registry import METRICS
from ..accuracy import accuracy
from ..iou import kaggle_iou


@METRICS.register("tgs_accuracy")
def tgs_accuracy(predicts, targets, threshold=0.5):
    return accuracy(predicts[0], targets[0], threshold)


@METRICS.register("tgs_kaggle_iou")
def tgs_kaggle_iou(predicts, targets):
    return kaggle_iou(predicts[0], targets[0])


@METRICS.register("tgs_kaggle_iou_v2")
def tgs_kaggle_iou_v2(predicts, targets):
    masks = predicts[0]
    labels = predicts[1]
    for i in range(len(masks)):
        if labels[i] <= 0.5:
            masks[i] = 0

    return kaggle_iou(masks, targets[0])


@METRICS.register("tgs_accurary_classify")
def tgs_accuracy_classify(predicts, targets, threshold=0.5):
    return accuracy(predicts[1], targets[1], threshold)
