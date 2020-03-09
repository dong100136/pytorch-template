from abc import abstractmethod, ABCMeta


class BaseHook(ABCMeta):
    name = 'BaseHook'

    @abstractmethod
    def __after_epoch_hook(self):
        pass

    @abstractmethod
    def __after_predict_hook(self):
        pass

    def __del__(self):
        return self.__end_hook()

    def __call__(self):
        return self.__predict_hook()
