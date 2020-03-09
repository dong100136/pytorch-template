from utils.registry import Registry

ARCH = Registry("ARCH")
LOSS = Registry("LOSS")
METRICS = Registry("METRICS")
OPTIMIZER = Registry("OPTIMIZER")
LR_SCHEDULER = Registry("LR_SCHEDULER")
TRAINER = Registry("TRAINER")
DATA_LOADER = Registry("DATA_LOADER")
HOOK = Registry("HOOK")
PREDICTOR = Registry("PREDICTOR")
