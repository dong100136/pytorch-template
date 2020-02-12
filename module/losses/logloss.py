import torch
import torch.nn.functional as F
from ..registry import LOSS

LOSS.register('nll_loss', F.nll_loss)

@LOSS.register('logloss')
def logloss(output, target):
    print(output.shape,target.shape)
    print(output[0])
    return F.nll_loss(torch.log(output), target)


class LogLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super(LogLoss, self).__init__()
        self.weights =torch.Tensor(weights).cuda()
        self.nll_loss = torch.nn.NLLLoss(weight=self.weights)

    def forward(self, input, target):
        return self.nll_loss(torch.log(input), target)
