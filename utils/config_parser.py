from module import *
import logging
import os
from functools import partial
from inspect import isfunction


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

        self.config['trainer']['args']['checkerpoint_dir'] = os.path.join(
            self.config['workspace'], 'model', self.config['exp_name']
        )
        self.config['trainer']['args']['tensorboard_dir'] = os.path.join(
            self.config['workspace'], 'log', self.config['exp_name']
        )
        self.config['prediction_path'] = os.path.join(
            self.config['workspace'], 'prediction', self.config['exp_name'],
            "prediction.csv"
        )

    def get_trainer(self, resume=True):
        return self.parser.get_trainer(resume)

    def init_dataloader(self, loader_name):
        return self.parser.init_dataloader(loader_name)

    def init_model(self):
        return self.parser.init_model()

    def get_hooks(self, hook_names):
        if self.config[hook_names] and len(self.config[hook_names])>0:
            return self.parser.get_hooks(hook_names)
        else:
            return []

    def _init_default_config_parser(self, config, logger):
        raise NotImplementedError

    def get_logger(self, name, verbosity=logging.NOTSET):
        logging.basicConfig(level=verbosity)
        logger = logging.getLogger(name)
        return logger

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, item, value):
        self.config[item] = value


class ConfigParserV2:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def get_trainer(self, resume=True):

        # init dataloader
        data_loader = self._init_obj(DATA_LOADER, self.config['dataloader'])
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = self._init_obj(ARCH, self.config['arch'])
        self.logger.info(model)

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

    def init_model(self):
        model = self._init_obj(ARCH, self.config['arch'])
        self.logger.info(model)
        return model

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

            module_args = dict(obj['args'])

            assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
            module_args.update(kwargs)

            obj = registry[module_name]
            if is_function:
                return partial(
                    obj, *args,**module_args
                )
            else:
                return obj(*args,**module_args)
            # return registry[module_name](*args,**module_args)
        elif isinstance(obj, str):
            return registry[obj]
