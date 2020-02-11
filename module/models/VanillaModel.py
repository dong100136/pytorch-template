import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from ..registry import ARCH
from module.basic_models.resnet import resnet50
from module.basic_models.inception import inception_v3
from module.basic_models.xception import xception


@ARCH.register("VanillaModel")
class VanillaModel(BaseModel):
     def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.resnet50 = resnet50(pretrained=pretrained, header=False)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 1))

        self.inceptionV3 = inception_v3(pretrained=pretrained, header=False)
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 1))

        self.xception = xception(pretrained=pretrained,header=False)
        self.pooling3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x1 = self.resnet50(x)
        x2 = self.inceptionV3(x)
        x3 = self.xception(x)

        print(x1.shape,x2.shape,x3.shape)

        return x
