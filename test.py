import argparse
import torch
from tqdm import tqdm
import numpy as np
import yaml
from yaml import FullLoader as Loader
from yaml import Dumper as Dumper
import pretty_errors
import logging
from pathlib import Path
import os

from utils.config_parser import ConfigParser


def to_device(data, device):
    if isinstance(data, list):
        data = [x.to(device, non_blocking=True) for x in data]
    elif isinstance(data, dict):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
    else:
        data = data.to(device, non_blocking=True)

    return data


class DataCollection:

    def __init__(self):
        self._data = []
        self.n_samples = 0

    def append(self, data):
        if not isinstance(data, (list, tuple)):
            data = [data]

        if len(self._data) == 0:
            for i in range(len(data)):
                self._data.append(data[i].detach())
        else:
            for i in range(len(data)):
                self._data[i] = torch.cat([self._data[i], data[i].detach()], dim=0)

        self.n_samples += len(data[0])

    def get_data(self):
        if len(self._data) == 1:
            return self._data[0]
        else:
            return self._data

    def __len__(self):
        return self.n_samples


def main(config, resume_model=None, device=None):
    logger = logging.getLogger('valid')

    configParser = ConfigParser(config)
    # setup data_loader instances
    data_loader = configParser.init_dataloader("valid_dataloader")
    model = configParser.init_model(verbose=False)

    predict_hooks = configParser.get_hooks('valid_hook')
    metrics = configParser.init_metrics('valid_metrics')

    best_model_path = configParser['trainer']['args']['checkerpoint_dir'] / "model_best.pth"
    best_model_path = Path(best_model_path)
    if resume_model == None and best_model_path.exists():
        print("find best model %s" % best_model_path)
        resume_model = best_model_path

    logger.info('Loading checkpoint: {} ...'.format(resume_model))
    checkpoint = torch.load(resume_model)
    state_dict = checkpoint['state_dict']
    if configParser['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    preds = DataCollection()
    targets = DataCollection()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            target = to_device(target, device)
            data = to_device(data, device)
            output = model(data)

            preds.append(output)
            targets.append(target)

    n_samples = len(data_loader.sampler)

    logger.info("=" * 50)
    logger.info("= support\t:%d" % (n_samples))
    params = {
        'predicts': preds.get_data(),
        'targets': targets.get_data()
    }
    for metric in metrics:
        score = metric(**params)
        logger.info("= %s\t:%f" % (metric.__name__, score))

    logger.info("=" * 50)

    params = {
        'dataset': data_loader.dataset,
        'predicts': preds,
        'targets': targets,
        'workspace': configParser['prediction_path']
    }

    for hook in predict_hooks:
        hook(**params)


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

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    main(config, args.resume, args.device)
