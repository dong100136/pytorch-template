import sys
from torchsummary import summary
from tensorwatch.model_graph.hiddenlayer.pytorch_draw_model import draw_img_classifier
import torch
import argparse
sys.path.append("/Users/stoneye/github/pytorch-template")

from module import ARCH

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

model = ARCH[args.model]('resnet34', 2)
# summary(model, ((3, 128, 128), (1)))
img = torch.rand(size=(2, 3, 128, 128))
labels = torch.rand(size=(2, 1))
fot = draw_img_classifier(model, [img, labels])
