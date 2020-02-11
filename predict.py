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


def main(config, resume_model=None, device=None):
    logger = logging.getLogger('predictor')

    configParser = ConfigParser(config)
    # setup data_loader instances
    data_loader = configParser.init_dataloader("predict_dataloader")
    model = configParser.init_model()

    predict_hooks = configParser.get_hooks('predict_hook')
    metrics = configParser.init_metrics()

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

    preds = None
    input_data = None
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if isinstance(data,list):
                data = data[0]
            data = data.to(device)
            output = model(data)

            output =  output.cpu().detach().numpy()
            if not isinstance(preds,np.ndarray):
                preds = output
            else:
                preds = np.vstack([preds, output])

    n_samples = len(data_loader.sampler)

    for hook in predict_hooks:
        hook(
            dataset = data_loader.dataset,
            predict = preds,
            workspace = configParser['prediction_path']
        )


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
