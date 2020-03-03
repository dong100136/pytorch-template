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

config = {
    'predictor_output': 'mask'
}


def to_device(data, device):
    if isinstance(data, list):
        data = [x.to(device, non_blocking=True) for x in data]
    elif isinstance(data, dict):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
    else:
        data = data.to(device, non_blocking=True)

    return data


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

    output_path = configParser['prediction_path'] / configParser['predictor_output']
    output_path.mkdir(exist_ok=True, parents=True)
    imgs_path = data_loader.dataset.imgs
    num_samples = len(imgs_path)
    i = 0

    with torch.no_grad():
        for data in tqdm(data_loader):
            if isinstance(data, list) and isinstance(data[0], list):
                data = data[0]
            data = to_device(data, device)
            output = model(data)

            masks = torch.sigmoid(output[0]).cpu().detach().numpy()

            for k in range(masks.shape[0]):
                img_path = output_path / ('%s.npy' % Path(imgs_path[i]).stem)
                if output[1][k] <= 0.5:
                    masks[k] = 0
                np.save(img_path, masks[k])
                i = i + 1
            # print("save output to %s" % img_path)


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
    main(config, args.resume, args.device)
