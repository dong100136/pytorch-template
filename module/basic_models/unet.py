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


class ResidualBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, down=False):
        super(ResidualBlock, self).__init__()
        strides = 2 if down else 1
        self.convReLu1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, strides, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.convReLu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, strides, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

        if in_channels == out_channels:
            self.identity = nn.Sequential(
                nn.Identity()
            )
        else:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, strides),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        y = self.convReLu1(x)
        y = self.convReLu2(y)
        y = y + self.identity(x)
        y = self.relu(y)
        return y


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, blocks):
        super(BasicBlock, self).__init__()
        residualBlocks = [ResidualBlock(in_channels, out_channels)]
        for i in range(blocks - 1):
            residualBlocks.append(ResidualBlock(out_channels, out_channels))

        self.layers = nn.ModuleList(residualBlocks)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.in_channels = [in_channels, 64, 128, 256]
        self.out_channels = [64, 128, 256, 512]
        self.blocks = [3, 4, 6, 3]
        self.layers = nn.ModuleList([
            BasicBlock(c1, c2, b)
            for c1, c2, b in zip(self.in_channels, self.out_channels, self.blocks)
        ])
        self.features = [Features(layer) for layer in self.layers]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.max_pool2d(x, 2, 2)
        return x


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
        self._encoder = Encoder(in_channels=3)
        self._features = self._encoder.features[::-1]
        self._skip_channels = self._encoder.out_channels[::-1]
        self._in_channels = [512, 512, 256, 128]  # ! 513 is depth feat
        self._out_channels = [512, 256, 128, 64]

        activation_fn = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

        self._upLayers = nn.ModuleList([
            UpLayer(c1, c2, c3)
            for c1, c2, c3 in zip(self._in_channels, self._skip_channels, self._out_channels)
        ])

        self._conv_activation = nn.Sequential(
            nn.Conv2d(self._out_channels[-1], num_classes, 3, padding=1),
            activation_fn
        )
        self.identity = nn.Identity()

        self.center = BasicBlock(513, 512, 2)

        self.aux = None
        if aux == 'classify':
            out_classes = 1 if aux_num_classes <= 2 else aux_num_classes
            activation_fn = nn.Sigmoid() if out_classes == 1 else nn.Softmax(dim=1)
            self.aux = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.classify = nn.Sequential(
                nn.Linear(512, out_classes),
                activation_fn
            )

    def forward(self, x):
        img, depth = x[0], x[1].float()
        depth_feat = depth.reshape((-1, 1, 1, 1)).expand(-1, -1, 8, 8)

        img_feat = self._encoder(img)
        img_feat = torch.cat([img_feat, depth_feat], dim=1)
        img_feat = self.center(img_feat)

        y = self.identity(img_feat)
        for uplayer, skip_x in zip(self._upLayers, self._features):
            y = uplayer(y, skip_x.features)

        y = self._conv_activation(y)
        y = y.squeeze(dim=1)

        if self.aux:
            aux_y = self.aux(img_feat)
            aux_y = self.classify(aux_y)
            aux_y = aux_y.squeeze(dim=1)
            return y, aux_y

        return y


@ARCH.register("UNet_v2")
class UNet_v2(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=2, pretrained=True, decoder_attention_type='scse'):
        super(UNet_v2, self).__init__()
        self.num_classes = num_classes
        weights = 'imagenet' if pretrained else None
        activation = 'sigmoid' if num_classes == 1 else 'softmax2d'

        aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )

        self._model = smp.Unet(backbone,
                               encoder_weights=weights,
                               classes=num_classes,
                               activation=activation,
                               in_channels=3,
                               decoder_attention_type=decoder_attention_type,
                               aux_params=aux_params)

    def forward(self, x):
        mask, label = self._model(x[0])
        if self.num_classes == 1:
            mask = mask.squeeze(dim=1)
            label = label.squeeze(dim=1)
        return mask, label
