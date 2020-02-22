import torch.nn.functional as F
from torch import nn
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.heads import SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base.modules import Activation
from ..registry import ARCH
from typing import Optional, Union, List


@ARCH.register("TGS_UNet")
class TGS_UNet(smp.base.SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):

        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        encoder_out_channels = list(self.encoder.out_channels)
        encoder_out_channels[-1] += 1

        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_out_channels[-1], **aux_params
            )

            if aux_params['pooling'] not in ("max", "avg"):
                raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(aux_params['pooling']))

            self.aux_pool = nn.AdaptiveAvgPool2d(1) if aux_params['pooling'] == 'avg' else nn.AdaptiveMaxPool2d(1)
            self.aux_flatten = nn.Flatten()

            self.classification_head = nn.Sequential(
                nn.Dropout(p=aux_params['dropout'], inplace=True),
                nn.Linear(encoder_out_channels[-1], aux_params['classes'], bias=True),
                nn.Sigmoid()
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        depth = x[1].float()
        depth_features = depth.reshape((-1, 1, 1, 1)).expand(-1, -1, 4, 4)

        img = x[0]
        features = self.encoder(img)
        features_without_depth = features[-1]

        features[-1] = torch.cat([features[-1], depth_features], dim=1)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        masks = masks.squeeze(dim=1)

        if self.classification_head is not None:
            features_without_depth = self.aux_pool(features_without_depth)
            features_without_depth = self.aux_flatten(features_without_depth)
            features_without_depth = torch.cat([features_without_depth, depth.reshape((-1, 1))], dim=1)

            labels = self.classification_head(features_without_depth)
            labels = labels.squeeze(dim=1)
            return masks, labels

        return masks
