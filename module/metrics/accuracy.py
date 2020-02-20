import torch
import numpy as np
from ..registry import METRICS


@METRICS.register("accuracy")
def accuracy(predicts, targets, threshold=0.5):
    with torch.no_grad():
        if len(predicts.shape) == 4:
            predict_labels = torch.argmax(predicts, dim=1)
        else:
            predict_labels = torch.zeros_like(predicts)
            predict_labels[predicts > threshold] = 1
        assert predict_labels.shape == targets.shape, "predicts: %s, targets:%s" % (
            predict_labels.shape, targets.shape)
        correct = 0
        correct += torch.sum(predict_labels == targets).item()

        total = 1
        for i in targets.shape:
            total *= i

        return correct / total
