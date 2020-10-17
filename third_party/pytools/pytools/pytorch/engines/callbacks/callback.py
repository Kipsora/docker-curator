from typing import Dict, Any, Optional, Callable
from pytools.pytorch.engines import Engine

__all__ = ['Callback', 'Lambda']


class Callback(object):
    def prior_all(self, engine: Engine):
        pass

    def after_all(self, engine: Engine):
        pass

    def prior_epoch(self, engine: Engine, data_loader):
        pass

    def after_epoch(self, engine: Engine, data_loader):
        pass

    def prior_batch(self, engine: Engine, inputs: Dict[str, Any]):
        pass

    def after_batch(self, engine: Engine, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        pass

    # Trainer only callback
    def prior_backward(self, engine: Engine):
        pass

    # Trainer only callback
    def after_backward(self, engine: Engine):
        pass

    def prior_forward(self, engine: Engine):
        pass

    def after_forward(self, engine: Engine):
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]):
        pass


class Lambda(Callback):
    def __init__(
            self,
            *,
            prior_all: Optional[Callable] = None,
            after_all: Optional[Callable] = None,
            prior_epoch: Optional[Callable] = None,
            after_epoch: Optional[Callable] = None,
            prior_batch: Optional[Callable] = None,
            after_batch: Optional[Callable] = None,
            prior_forward: Optional[Callable] = None,
            after_forward: Optional[Callable] = None,
            prior_backward: Optional[Callable] = None,
            after_backward: Optional[Callable] = None):
        self._prior_all_fn = prior_all
        self._after_all_fn = after_all
        self._prior_epoch_fn = prior_epoch
        self._after_epoch_fn = after_epoch
        self._prior_batch_fn = prior_batch
        self._after_batch_fn = after_batch
        self._prior_forward_fn = prior_forward
        self._after_forward_fn = after_forward
        self._prior_backward_fn = prior_backward
        self._after_backward_fn = after_backward

    def prior_all(self, engine: Engine):
        if self._prior_all_fn:
            self._prior_all_fn(engine)

    def after_all(self, engine: Engine):
        if self._after_all_fn:
            self._after_all_fn(engine)

    def prior_epoch(self, engine: Engine, data_loader):
        if self._prior_epoch_fn:
            self._prior_epoch_fn(engine, data_loader)

    def after_epoch(self, engine: Engine, data_loader):
        if self._after_epoch_fn:
            self._after_epoch_fn(engine, data_loader)

    def prior_batch(self, engine: Engine, inputs: Dict[str, Any]):
        if self._prior_batch_fn:
            self._prior_batch_fn(engine, inputs)

    def after_batch(self, engine: Engine, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        if self._after_batch_fn:
            self._after_batch_fn(engine, inputs, outputs)

    def prior_forward(self, engine: Engine):
        if self._prior_forward_fn:
            self._prior_forward_fn(engine)

    def after_forward(self, engine: Engine):
        if self._after_forward_fn:
            self._after_forward_fn(engine)

    def prior_backward(self, engine: Engine):
        if self._prior_backward_fn:
            self._prior_backward_fn(engine)

    def after_backward(self, engine: Engine):
        if self._after_backward_fn:
            self._after_backward_fn(engine)
