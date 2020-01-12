import torch
import torch.nn.functional as F


def logloss(output, target):
    return F.nll_loss(torch.log(output), target)


class LogLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super(LogLoss, self).__init__()
        self.weights =torch.Tensor(weights).cuda()
        self.nll_loss = torch.nn.NLLLoss(weight=self.weights)

    def forward(self, input, target):
        return self.nll_loss(torch.log(input), target)
