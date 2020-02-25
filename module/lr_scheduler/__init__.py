import torch
from ..registry import LR_SCHEDULER

LR_SCHEDULER.register('StepLR', torch.optim.lr_scheduler.StepLR)
LR_SCHEDULER.register('ReduceLROnPlateau', torch.optim.lr_scheduler.ReduceLROnPlateau)
LR_SCHEDULER.register("CyclicLR", torch.optim.lr_scheduler.CyclicLR)
