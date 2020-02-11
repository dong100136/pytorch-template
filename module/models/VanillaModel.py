import torch
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

        self.xception = xception(pretrained=pretrained, header=False)
        self.pooling3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(2048*3, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x1 = self.resnet50(x)
        x2,_ = self.inceptionV3(x)
        x3 = self.xception(x)


        x1 = self.pooling1(x1)
        x2 = self.pooling2(x2)
        x3 = self.pooling3(x3)

        x1 = torch.flatten(x1,start_dim=1)
        x2 = torch.flatten(x2,start_dim=1)
        x3 = torch.flatten(x3,start_dim=1)

        x = torch.cat([x1,x2,x3], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
