import torch
import torch.nn as nn
import torch.nn.functional as F
from ..DiceLoss import DiceLoss, BinaryDiceLoss
from ..Lovasz import lovasz_hinge
from ..logloss import bce
from ...registry import LOSS


@LOSS.register('Tgs_Loss')
class Tgs_Loss(nn.Module):
    def __init__(self):
        super(Tgs_Loss, self).__init__()
        self.diceLoss = BinaryDiceLoss()

    def forward(self, preds, targets):
        pred_masks, pred_labels, upsample_masks = preds
        target_masks, target_labels = targets

        pred_masks = torch.sigmoid(pred_masks)
        upsample_masks = [torch.sigmoid(x) for x in upsample_masks]

        # loss1 = self.diceLoss(pred_masks, target_masks)
        loss2 = F.binary_cross_entropy_with_logits(pred_masks, target_masks.float())
        loss3 = F.binary_cross_entropy_with_logits(pred_labels, target_labels.float())

        for mask in upsample_masks:
            mask = mask.squeeze(1)
            loss2 += F.binary_cross_entropy_with_logits(mask, target_masks.float())

        loss2 = loss2 / (len(upsample_masks) + 1)

        loss = 0.8 * loss2 + 0.2 * loss3
        # loss = 0.8 * loss1 + 0.2 * loss3

        return loss.mean()


def symmetric_lovasz(outputs, targets):
    return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2


# def symmetric_lovasz_ignore_empty(outputs, targets, truth_image):
#     return (lovasz_loss_ignore_empty(outputs, targets, truth_image)
#             + lovasz_loss_ignore_empty(-outputs, 1 - targets, truth_image)) / 2


@LOSS.register('Tgs_Loss_v2')
class Tgs_Loss_v2(nn.Module):
    def __init__(self):
        super(Tgs_Loss_v2, self).__init__()

    def forward(self, preds, targets):
        pred_masks, pred_labels, upsample_masks = preds
        target_masks, target_labels = targets

        assert target_masks.shape == pred_masks.shape

        # loss1 = lovasz_hinge(pred_masks, target_masks)

        loss = symmetric_lovasz(pred_masks, target_masks)
        for upsample_mask in upsample_masks:
            loss += symmetric_lovasz(upsample_mask, target_masks)

        loss += F.binary_cross_entropy_with_logits(pred_labels, target_labels.float(), reduction='mean')

        # loss3 = F.binary_cross_entropy_with_logits(pred_labels, target_labels.float())

        # loss = 0.8 * loss1 + 0.2 * loss3
        # loss = loss1

        return loss
