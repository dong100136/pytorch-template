import torch
import torch.nn.functional as F
from ..registry import METRICS

@METRICS.register("logloss")
def logloss(output, target, dim=1):
    with torch.no_grad():
        loss = F.nll_loss(torch.log(output),target)

    return loss
