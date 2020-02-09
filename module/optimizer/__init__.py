from ..registry import OPTIMIZER
from torch.optim import Adam

OPTIMIZER.register('Adam', Adam)