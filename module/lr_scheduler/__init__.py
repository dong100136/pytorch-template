import torch
from ..registry import LR_SCHEDULER

LR_SCHEDULER.register('StepLR', torch.optim.lr_scheduler.StepLR)