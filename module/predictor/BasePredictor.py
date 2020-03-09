from abc import abstractmethod, ABC
from pathlib import Path
import torch
import logging
from utils import to_device
from tqdm import tqdm


class BasePredictor(ABC):
    def __init__(self, config, model, dataloader, hooks, **kwargs):
        self.logger = logging.getLogger('predictor')
        self.config = config
        self.predict_hooks = hooks
        self.dataloader = dataloader
        self.hooks = hooks
        self.model = model

    def _load_best_model(self, resume_model=None):
        best_model_path = self.config['trainer']['args']['checkerpoint_dir'] / "model_best.pth"
        best_model_path = Path(best_model_path)
        if resume_model == None and best_model_path.exists():
            print("find best model %s" % best_model_path)
            resume_model = best_model_path

        self.logger.info('Loading checkpoint: {} ...'.format(resume_model))
        checkpoint = torch.load(resume_model)
        state_dict = checkpoint['state_dict']
        if self.config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(model)
        self.model.load_state_dict(state_dict)

        # prepare model for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self):
        self._load_best_model()
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                if isinstance(data, list) and isinstance(data[0], list):
                    data = data[0]
                data = to_device(data, self.device)
                result = self.predict_epoch(self.model, data)
                result = result.detach().cpu()
                self._after_epoch(result)

        self._after_predict()

    @abstractmethod
    def predict_epoch(self, model, data):
        """[prediction process for one epoch]

        Arguments:
            data {[torch.tensor]} -- [data for one epoch]
        """
        pass

    def _after_epoch(self, result):
        for hook in self.hooks:
            hook.after_epoch_hook(result)

    def _after_predict(self):
        for hook in self.hooks:
            del hook
