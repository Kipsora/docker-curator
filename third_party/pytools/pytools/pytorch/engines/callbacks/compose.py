from typing import Any, Dict, Collection

from pytools.pytorch.engines.callbacks import Callback

__all__ = ['Compose']


class Compose(Callback):
    def __init__(self, callbacks: Collection[Callback]):
        self._callbacks = callbacks

    def prior_batch(self, engine, inputs):
        for callback in self._callbacks:
            callback.prior_batch(engine, inputs)

    def after_batch(self, engine, inputs, outputs):
        for callback in self._callbacks:
            callback.after_batch(engine, inputs, outputs)

    def prior_epoch(self, engine, data_loader):
        for callback in self._callbacks:
            callback.prior_epoch(engine, data_loader)

    def after_epoch(self, engine, data_loader):
        for callback in self._callbacks:
            callback.after_epoch(engine, data_loader)

    def prior_all(self, engine):
        for callback in self._callbacks:
            callback.prior_all(engine)

    def after_all(self, engine):
        for callback in self._callbacks:
            callback.after_all(engine)

    def prior_forward(self, engine):
        for callback in self._callbacks:
            callback.prior_forward(engine)

    def after_forward(self, engine):
        for callback in self._callbacks:
            callback.after_forward(engine)

    def prior_backward(self, engine):
        for callback in self._callbacks:
            callback.prior_backward(engine)

    def after_backward(self, engine):
        for callback in self._callbacks:
            callback.after_backward(engine)

    def state_dict(self) -> Dict[str, Any]:
        state = super(Compose, self).state_dict()
        state.setdefault('compose', [callback.state_dict() for callback in self._callbacks])
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        super(Compose, self).load_state_dict(state)
        if 'compose' in state:
            assert len(state['compose']) == len(self._callbacks)
            for callback, callback_state in zip(self._callbacks, state['compose']):
                callback.load_state_dict(callback_state)
