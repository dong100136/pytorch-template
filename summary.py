import argparse
import torch
from tqdm import tqdm
import numpy as np
import yaml
from yaml import FullLoader as Loader
from yaml import Dumper as Dumper
import pretty_errors
import logging
from torchsummary import summary

from utils.config_parser import ConfigParser


def main(config):
    logger = logging.getLogger('predictor')

    configParser = ConfigParser(config)
    # setup data_loader instances
    model = configParser.init_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, (3, 224, 224))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    main(config)
