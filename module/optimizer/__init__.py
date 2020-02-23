from ..registry import OPTIMIZER
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW

OPTIMIZER.register('Adam', Adam)
OPTIMIZER.register('SGD', SGD)
OPTIMIZER.register('AdamW', AdamW)
