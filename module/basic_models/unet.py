import torch.nn.functional as F
from torch import nn
import torch
import segmentation_models_pytorch as smp
from ..registry import ARCH


class ResNetDecoder(nn.Module):
    def __init__(self, backbone, *args, **kwargs):
        super(ResNetDecoder, self).__init__()
        self._model = ARCH[backbone](*args, **kwargs)
        back_models = list(self._model.named_children())
        self._selected_layers = [back_models[x][1] for x in [2, 4, 5, 6]]
        self._identity = nn.Identity()
        self.features = [Features(self._identity)] + [Features(layer) for layer in self._selected_layers]
        self._channels = [3, 64, 64, 128, 256]

    def forward(self, x):
        x = self._identity(x)
        return self._model(x)


class Features:
    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, x_in, x_out):
        self.features = x_out

    def remove(self):
        self.hook.remove()


class UpLayer(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpLayer, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv_relu1 = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip_x):
        x = self.transpose_conv(x)
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv_relu1(x)
        x = self.conv_relu2(x)
        return x


@ARCH.register("UNet")
class UNet(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=10,
                 pretrained=True,
                 aux="none", aux_num_classes=2):
        """[summary]

        Keyword Arguments:
            backbone {str} -- [description] (default: {'resnet34'})
            num_classes {int} -- [description] (default: {10})
            pretrained {bool} -- [description] (default: {True})
            aux {str} -- aux=[classify|regression| none] (default: {"none"})
        """
        super(UNet, self).__init__()
        self._encoder = ResNetDecoder(backbone=backbone, pretrained=pretrained, header=False)
        self._features = self._encoder.features[::-1]
        self._skip_channels = self._encoder._channels[::-1]
        self._up_channels = [512, 256, 128, 64, 64]
        self._out_channels = [256, 128, 64, 64, 32]

        activation_fn = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

        self._upLayers = nn.ModuleList([
            UpLayer(c1, c2, c3)
            for c1, c2, c3 in zip(self._up_channels, self._skip_channels, self._out_channels)
        ])

        self._conv_activation = nn.Sequential(
            nn.Conv2d(32, num_classes, 3, padding=1),
            activation_fn
        )
        self.identity = nn.Identity()

        self.aux = None
        if aux == 'classify':
            out_classes = 1 if aux_num_classes <= 2 else aux_num_classes
            activation_fn = nn.Sigmoid() if out_classes == 1 else nn.Softmax(dim=1)
            self.aux = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, out_classes),
                activation_fn
            )

    def forward(self, x):

        x = self._encoder(x[0])
        print(x.shape)

        y = self.identity(x)
        for uplayer, skip_x in zip(self._upLayers, self._features):
            y = uplayer(y, skip_x.features)

        y = self._conv_activation(y)
        y = y.squeeze(dim=1)

        if self.aux:
            aux_y = self.aux(x)
            aux_y = aux_y.squeeze(dim=1)
            return y, aux_y

        return y


@ARCH.register("UNet_v2")
class UNet_v2(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=2, pretrained=True):
        super(UNet_v2, self).__init__()
        self.num_classes = num_classes
        weights = 'imagenet' if pretrained else None
        activation = 'sigmoid' if num_classes == 1 else 'softmax2d'
        self._model = smp.Unet(backbone,
                               encoder_weights=weights,
                               classes=num_classes,
                               activation=activation,
                               in_channels=3)

    def forward(self, x):
        x = self._model(x)
        if self.num_classes == 1:
            x = x.squeeze(dim=1)
        return x
