import torch
import torch.nn as nn
import torch.nn.functional as F
from ..DiceLoss import DiceLoss, BinaryDiceLoss
from ..logloss import bce
from ...registry import LOSS


@LOSS.register('Tgs_Loss')
class Tgs_Loss(nn.Module):
    def __init__(self):
        super(Tgs_Loss, self).__init__()
        self.diceLoss = BinaryDiceLoss()

    def forward(self, preds, targets):
        pred_masks, pred_labels = preds
        target_masks, target_labels = targets

        loss1 = self.diceLoss(pred_masks, target_masks)
        loss2 = F.binary_cross_entropy_with_logits(pred_masks, target_masks.float())
        loss3 = F.binary_cross_entropy_with_logits(pred_labels, target_labels.float())

        loss = 0.3 * loss1 + 0.4 * loss2 + 0.3 * loss3

        return loss.mean()
