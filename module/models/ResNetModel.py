import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from module.basic_models.resnet import resnet50


class ResNetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet50(pretrained=True, num_classes=10)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.model(x)
        x = torch.squeeze(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = F.softmax(x, dim=-1)
        return x
