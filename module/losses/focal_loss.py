import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = torch.nn.NLLLoss()

    def forward(self, input, target):
        logp = self.nll_loss(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()