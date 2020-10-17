import abc
from typing import Optional, Any, Dict

from pytools.pytorch.typing import Callback

__all__ = ['Engine']


class Engine(object, metaclass=abc.ABCMeta):
    def __init__(self, callback: Optional[Callback]):
        self._callback = callback

    @property
    @abc.abstractmethod
    def global_step(self):
        raise NotImplementedError

    @property
    def callback(self):
        return self._callback

    def _prior_all(self):
        if self._callback is not None:
            self._callback.prior_all(self)

    def _after_all(self):
        if self._callback is not None:
            self._callback.after_all(self)

    def _prior_batch(self, inputs):
        if self._callback is not None:
            self._callback.prior_batch(self, inputs)

    def _after_batch(self, inputs, outputs):
        if self._callback is not None:
            self._callback.after_batch(self, inputs, outputs)

    def _prior_epoch(self, data_loader):
        if self._callback is not None:
            self._callback.prior_epoch(self, data_loader)

    def _after_epoch(self, data_loader):
        if self._callback is not None:
            self._callback.after_epoch(self, data_loader)

    def _prior_forward(self):
        if self._callback is not None:
            self._callback.prior_forward(self)

    def _after_forward(self):
        if self._callback is not None:
            self._callback.after_forward(self)

    def _prior_backward(self):
        if self._callback is not None:
            self._callback.prior_backward(self)

    def _after_backward(self):
        if self._callback is not None:
            self._callback.after_backward(self)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'callback': self._callback.state_dict()
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if 'callback' in state:
            self._callback.load_state_dict(state['callback'])
