import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from module.basic_models.densenet import densenet161


class DenseNetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = densenet161(pretrained=True, header=False)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(self.model.num_features, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x)
        x = torch.flatten(x,1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = F.softmax(x, dim=-1)
        return x
