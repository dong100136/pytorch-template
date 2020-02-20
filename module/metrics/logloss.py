import torch
import torch.nn.functional as F
from ..registry import METRICS


@METRICS.register("logloss")
def logloss(predicts, targets, dim=1):
    with torch.no_grad():
        loss = F.nll_loss(torch.log(predicts), targets)

    return loss
