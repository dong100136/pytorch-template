import segmentation_models_pytorch as smp
from ..registry import ARCH
import torch


@ARCH.register("FPN")
def FPN(num_classes=2, backbone='resnet34', in_channels=3):
    model = smp.FPN(backbone, in_channels=in_channels,
                    classes=num_classes,
                    activation='softmax2d',
                    encoder_weights='imagenet')
    return model
