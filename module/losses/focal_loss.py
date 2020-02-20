import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSS


@LOSS.register('FocalLoss')
class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        logp = F.nll_loss(torch.log(input), target, reduction='none')
        p = torch.exp(-logp)

        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
