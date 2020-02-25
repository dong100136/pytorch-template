from module import *
import logging
import os
from functools import partial
from inspect import isfunction
from pathlib import Path


class ConfigParser:
    def __init__(self, config):
        logger = self.get_logger('train')
        logger.warning("test")

        self.config = config
        self.version = config['version']

        if self.version == 2:
            self.parser = ConfigParserV2(config, logger)
        else:
            self.parser = self._init_default_config_parser(config, logger)

        self.config['trainer']['args']['checkerpoint_dir'] = Path(
            self.config['workspace']) / 'model' / self.config['exp_name']

        self.config['trainer']['args']['tensorboard_dir'] = Path(
            self.config['workspace']) / 'log' / self.config['exp_name']

        self.config['prediction_path'] = Path(
            self.config['workspace']) / 'prediction' / self.config['exp_name']

        self.config['trainer']['args']['checkerpoint_dir'].mkdir(parents=True, exist_ok=True)
        self.config['trainer']['args']['tensorboard_dir'].mkdir(parents=True, exist_ok=True)
        self.config['prediction_path'].mkdir(parents=True, exist_ok=True)

    def get_trainer(self, resume=None):
        return self.parser.get_trainer(resume)

    def init_dataloader(self, loader_name):
        return self.parser.init_dataloader(loader_name)

    def init_model(self, verbose=True):
        return self.parser.init_model(verbose=verbose)

    def init_metrics(self, name='metrics'):
        if self.config[name] and len(self.config[name]) > 0:
            return self.parser.init_metrics(name)
        else:
            return []

    def get_hooks(self, hook_names):
        if hook_names in self.config and self.config[hook_names] and len(self.config[hook_names]) > 0:
            return self.parser.get_hooks(hook_names)
        else:
            return []

    def _init_default_config_parser(self, config, logger):
        raise NotImplementedError

    def get_logger(self, name, verbosity=logging.INFO):
        logging.basicConfig(level=verbosity)
        logger = logging.getLogger(name)
        return logger

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, item, value):
        self.config[item] = value


class ConfigParserV2:
    # this is for default config
    config = {
        'lr_scheduler': {
            'type': 'ReduceLROnPlateau'
        },
        'optimizer': {
            'type': 'Adam',
            'args': {
                'lr': float('1e-4')
            }
        }
    }

    def __init__(self, config, logger):
        self.config.update(config)
        self.logger = logger

        lr = self.config['optimizer']['args']['lr']
        if isinstance(lr, str):
            self.config['optimizer']['args']['lr'] = float(lr)

        self.config['trainer']['args']['n_gpu'] = self.config['n_gpu']

    def __save_model_arch_to_file(self, model, file_path):
        with open(file_path / "model.arch", 'w') as f:
            f.write(str(model))

        self.logger.info("save model arch to %s" % (file_path / "model.arch"))

    def get_trainer(self, resume=None):

        # init dataloader
        data_loader = self._init_obj(DATA_LOADER, self.config['dataloader'])
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = self._init_obj(ARCH, self.config['arch'])
        self.__save_model_arch_to_file(model, self.config['trainer']['args']['checkerpoint_dir'])

        # get function handles of loss and metrics
        criterion = self._init_obj(LOSS, self.config['loss'])

        metrics = [self._init_obj(METRICS, met) for met in self.config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = self._init_obj(OPTIMIZER, self.config['optimizer'], trainable_params)

        lr_scheduler = self._init_obj(LR_SCHEDULER, self.config['lr_scheduler'], optimizer)

        self.logger.info(self.config['trainer']['args'])
        self.config['trainer']['args']['resume'] = resume
        trainer = self._init_obj(TRAINER, self.config['trainer'],
                                 model, criterion, metrics, optimizer,
                                 config=self.config,
                                 data_loader=data_loader,
                                 valid_data_loader=valid_data_loader,
                                 lr_scheduler=lr_scheduler,
                                 logger=self.logger)

        return trainer

    def init_model(self, verbose=True):
        model = self._init_obj(ARCH, self.config['arch'])
        if verbose:
            self.logger.info(model)
        return model

    def init_metrics(self, name='metrics'):
        return [
            self._init_obj(METRICS, metric)
            for metric in self.config[name]
        ]

    def init_dataloader(self, dataloader_name):
        dataloader = self._init_obj(DATA_LOADER, self.config[dataloader_name])
        return dataloader

    def get_hooks(self, hook_names):
        return [
            self._init_obj(HOOK, hook_name)
            for hook_name in self.config[hook_names]
        ]

    def _init_obj(self, registry, obj, *args, **kwargs):
        if isinstance(obj, dict):
            is_function = ('func' in obj)
            if is_function:
                module_name = obj['func']
            else:
                module_name = obj['type']

            if 'args' in obj:
                module_args = dict(obj['args'])
            else:
                module_args = {}

            assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
            module_args.update(kwargs)

            obj = registry[module_name]
            if is_function:
                return partial(
                    obj, *args, **module_args
                )
            else:
                return obj(*args, **module_args)
            # return registry[module_name](*args,**module_args)
        elif isinstance(obj, str):
            return registry[obj]
