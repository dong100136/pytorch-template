import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pretty_errors
import torch
import yaml
from tqdm import tqdm
from yaml import Dumper as Dumper
from yaml import FullLoader as Loader

from ..registry import PREDICTOR
from .BasePredictor import BasePredictor


@PREDICTOR.register('SegPredictor')
class SegPredictor(BasePredictor):
    def __init__(self, **kwargs):
        super(SegPredictor, self).__init__(**kwargs)

    def predict_epoch(self, model, data):
        output = model(data)
        return output
