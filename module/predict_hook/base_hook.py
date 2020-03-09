from abc import abstractmethod, ABC


class BaseHook(ABC):
    name = 'BaseHook'

    @abstractmethod
    def after_epoch_hook(self, predicts, targets=None, **kwargs):
        pass

    @abstractmethod
    def after_predict_hook(self):
        pass

    def __del__(self):
        self.after_predict_hook()

    def __call__(self, **kwargs):
        return self.after_epoch_hook(**kwargs)
