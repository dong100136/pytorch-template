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
import numpy as np

# ！ this is a bug
torch.backends.cudnn.enabled = False


class Features:
    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, x_in, x_out):
        self.features = x_out

    def remove(self):
        self.hook.remove()


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
        out_size: List[int] = (128, 128)
    ):
        super().__init__()
        self.out_size = out_size

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # change the arch of encoder
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, 3, 2, padding=1)
        self.encoder.maxpool = nn.Identity()

        encoder_out_channels = list(self.encoder.out_channels)
        encoder_out_channels[-1] += 1

        #! skip one downsample
        encoder_depth = 4
        decoder_channels = (256, 128, 64, 32)
        encoder_out_channels[2] += encoder_out_channels[1]
        encoder_out_channels[1] = 3

        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.decoder_features = [Features(x) for x in self.decoder.blocks]
        self.decoder_features_channels = [encoder_out_channels[-1]] + list(decoder_channels)

        self.upsample = nn.ModuleList([
            nn.UpsamplingBilinear2d(size=out_size)
            for _ in self.decoder_features_channels
        ])
        self.upsample_conv = nn.ModuleList([
            self.conv2d_bn_activate_conv2d(in_c, classes)
            for in_c in self.decoder_features_channels
        ])

        self.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(np.sum(self.decoder_features_channels), 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 1, 3, padding=1)
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

    def conv2d_bn_activate_conv2d(self, in_channels, out_channels, filters=32):
        return nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(in_channels, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, out_channels, 3, padding=1)
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        depth = x[1].float()

        img = x[0]
        features = self.encoder(img)
        features_without_depth = features[-1]
        h, w = features_without_depth.shape[2:]

        depth_features = depth.reshape((-1, 1, 1, 1)).expand((-1, -1, h, w))
        encoder_feature = torch.cat([features[-1], depth_features], dim=1)
        features[-1] = encoder_feature

        #! skip one downsample
        features[2] = torch.cat([features[1], features[2]], dim=1)
        features[1] = features[0]
        decoder_output = self.decoder(*features)

        # * get decoder features
        decoder_features = [encoder_feature, ] + [x.features for x in self.decoder_features]
        upsample_features = [
            layer(x)
            for layer, x in zip(self.upsample, decoder_features)
        ]

        upsample_masks = [
            layer(x) for layer, x in zip(self.upsample_conv, upsample_features)
        ]

        concat_features = torch.cat(upsample_features, dim=1)

        masks = self.segmentation_head(concat_features)
        masks = masks.squeeze(dim=1)

        if self.classification_head is not None:
            features_without_depth = self.aux_pool(features_without_depth)
            features_without_depth = self.aux_flatten(features_without_depth)
            features_without_depth = torch.cat([features_without_depth, depth.reshape((-1, 1))], dim=1)

            labels = self.classification_head(features_without_depth)
            labels = labels.squeeze(dim=1)
            return masks, labels, upsample_masks
            # return masks, labels

        return masks
