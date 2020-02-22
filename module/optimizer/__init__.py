from ..registry import OPTIMIZER
from torch.optim import Adam
from torch.optim import SGD

OPTIMIZER.register('Adam', Adam)
OPTIMIZER.register('SGD', SGD)
