import torch
import torch.nn.functional as F
from ..registry import LOSS

LOSS.register('nll_loss', F.nll_loss)


@LOSS.register('logloss')
def logloss(output, target):
    return F.nll_loss(torch.log(output), target)


@LOSS.register("BCEWithLogitsLoss")
def bce(output, target, *args, **kwargs):
    target = target.float()
    return F.binary_cross_entropy_with_logits(output, target, *args, **kwargs)


class LogLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super(LogLoss, self).__init__()
        self.weights = torch.Tensor(weights).cuda()
        self.nll_loss = torch.nn.NLLLoss(weight=self.weights)

    def forward(self, input, target):
        return self.nll_loss(torch.log(input), target)
