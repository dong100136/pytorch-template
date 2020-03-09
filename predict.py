import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pretty_errors
import torch
from tqdm import tqdm

from utils.config_parser import ConfigParser


def main(config, resume_model=None, device=None):
    logger = logging.getLogger('predictor')
    config_parser = ConfigParser(config)
    predictor = config_parser.get_predicter()

    predictor.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.config, args.resume, args.device)
