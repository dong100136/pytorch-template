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
from ...utils.utils import to_device
from ...utils.config_parser import ConfigParser
from abc import overide, abstractmethod

config = {
    'predictor_output': 'mask'
}


class BasePredictor:
    def __init__(self, config):
        self.logger = logging.getLogger('predictor')
        self.configParser = ConfigParser(config)
        self.data_loader = self.configParser.init_dataloader("predictor_dataloader")

        self.predictor_hooks = self.config.get_hooks()
        self.end_hooks = self.config.get_hooks()

    def __load_best_model(self, resume_model=None):
        self.model = self.configParser.init_model()

        best_model_path = self.configParser['trainer']['args']['checkerpoint_dir'] / "model_best.pth"
        best_model_path = Path(best_model_path)
        if resume_model == None and best_model_path.exists():
            print("find best model %s" % best_model_path)
            resume_model = best_model_path

        self.logger.info('Loading checkpoint: {} ...'.format(resume_model))
        checkpoint = torch.load(resume_model)
        state_dict = checkpoint['state_dict']
        if self.configParser['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(model)
        self.model.load_state_dict(state_dict)

        # prepare model for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self):
        self.__load_best_model()
        with torch.no_grad():
            for data in tqdm(self.data_loader):
                if isinstance(data, list) and isinstance(data[0], list):
                    data = data[0]
                data = to_device(data, device)
                result = self.__predict(data)
                self.__predict_hook(result)

    @abstractmethod
    def __predict(self, data):
        pass

    def __predict_hook(self, result):

    def __end_hook(self):
        pass


class SegPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        output_path = self.configParser['prediction_path'] / self.configParser['predictor_output']
        output_path.mkdir(exist_ok=True, parents=True)

        imgs_path = data_loader.dataset.imgs
        num_samples = len(imgs_path)
        i = 0

    @overide
    def __predict(self, data):
        output = model(data)
        return output


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
        tmp_config = yaml.load(f, Loader=Loader)
        config.update(tmp_config)
    main(args.config, args.resume, args.device)
