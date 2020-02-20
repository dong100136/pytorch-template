import torch
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    with torch.no_grad():
        b, h, w = np.array(input.shape)
        shape = (b, num_classes, h, w)
        result = torch.zeros(shape)
        result = result.scatter_(1, input.view(b, 1, h, w).cpu(), 1).long().cuda()

        return result
