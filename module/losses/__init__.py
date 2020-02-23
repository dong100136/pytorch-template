from .logloss import logloss, LogLoss
from torch.nn.functional import cross_entropy
from .focal_loss import FocalLoss
from .GHM import GHMC_Loss, GHMR_Loss
from .DiceLoss import DiceLoss
from .kaggle.tgs_loss import *
