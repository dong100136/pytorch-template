import torch
import numpy as np
from ..registry import METRICS

@METRICS.register("accuracy")
def accuracy(output, target):
    if isinstance(output,np.ndarray):
        pred = np.argmax(output,axis=1)
        assert pred.shape[0] ==len(target)
        correct = 0
        correct += np.sum(pred==target).item()
    else:
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()

    return correct / len(target)
