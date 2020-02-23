import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import os
from pathlib import Path
import logging
import pandas as pd


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, logger, **kwargs):
        self.config = config
        self.logger = logging.getLogger()
        cfg_trainer = config['trainer']['args']

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(cfg_trainer['n_gpu'])
        self.model = model.to(self.device)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = Path(cfg_trainer['checkerpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_last_models = 5

        # setup visualization writer instance
        self.writer = TensorboardWriter(cfg_trainer['tensorboard_dir'],
                                        self.logger,
                                        cfg_trainer['tensorboard'])

        if cfg_trainer['resume'] is not None:
            self.load_pretrain_model(cfg_trainer['resume'])

        self.load_last_checkpoint()

    def load_pretrain_model(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading pretrain model: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def load_last_checkpoint(self):
        max_epoch = -1
        max_epoch_pth_path = None
        # find last checkpoint from model_dir
        for checkpoint_pth in self.checkpoint_dir.glob("*.pth"):
            pth_name = checkpoint_pth.stem
            if pth_name.find('epoch') >= 0:
                epoch_num = int(pth_name.split("epoch")[-1])
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    max_epoch_pth_path = checkpoint_pth

        if max_epoch > 0:
            self.logger.info(
                "find latest checkpoint {} and load it.".format(max_epoch_pth_path.stem))
            self._resume_checkpoint(max_epoch_pth_path)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # self.writer.add_scalar('lr',self.lr,epoch)
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            self.logger.info('=' * 50)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            self.logger.info('=' * 50)

            # save loss and val_loss
            if (self.checkpoint_dir / 'loss.log').exists():
                save_loss = pd.read_csv(self.checkpoint_dir / 'loss.log')
            else:
                save_loss = pd.DataFrame({
                    'epoch': [],
                    'loss': [],
                    'val_loss': []
                })

            if epoch in save_loss['epoch']:
                print('1')
                save_loss.loc[save_loss['epoch'] == epoch, 'loss'] = log['loss']
                save_loss.loc[save_loss['epoch'] == epoch, 'val_loss'] = log['val_loss']
            else:
                d = pd.Series({'epoch': epoch, 'loss': log['loss'], 'val_loss': log['val_loss']})
                save_loss = save_loss.append(d, ignore_index=True)
            save_loss.to_csv(self.checkpoint_dir / 'loss.log', index=None)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self.logger.info("Congratulation, the training is finish!")

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

        # delete old pth
        for pth_file in self.checkpoint_dir.glob("*.pth"):
            pth_name = pth_file.stem
            if pth_name.find('epoch') >= 0:
                old_epoch = int(pth_name.split("epoch")[-1])
                if epoch - old_epoch > self.save_last_models:
                    self.logger.info("removing older pth {}".format(pth_name))
                    os.remove(pth_file)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch']['type'] != self.config['arch']['type']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
