import sys
from torchsummary import summary
import torch
import argparse
sys.path.append("/Users/stoneye/github/pytorch-template")

from module import ARCH

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

model = ARCH[args.model]('resnet34', 2)
summary(model, (3, 148, 148))
