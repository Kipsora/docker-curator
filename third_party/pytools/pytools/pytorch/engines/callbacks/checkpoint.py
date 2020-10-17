from typing import Any, Dict

from pytools.pytorch import distributed
from pytools.pytorch.engines import Trainer
from pytools.pytorch.engines.callbacks import Callback
from pytools.pytorch.session import SessionManager
from pytools.pytorch.summary import Summary

__all__ = ['SaveOnBestBenchmark', 'SaveOnEveryNEpochs']


class SaveOnBestBenchmark(Callback):
    def __init__(self, summary: Summary, target_field, *, manager: SessionManager, logger):
        self._summary = summary
        self._target_field = target_field

        self._best_value = None
        self._logger = logger
        self._manager = manager

    def after_epoch(self, engine: Trainer, data_loader):
        if distributed.is_local_master():
            current = self._summary[self._target_field].get_value(global_step=engine.global_step)
            if self._best_value is None or self._best_value < current:
                self._best_value = current
                self._manager.save(engine.state_dict(), global_step=engine.global_step, logger=self._logger)

    def state_dict(self) -> Dict[str, Any]:
        state = super(SaveOnBestBenchmark, self).state_dict()
        state.update({
            'best_value': self._best_value
        })
        return state
        
    def load_state_dict(self, state: Dict[str, Any]):
        super(SaveOnBestBenchmark, self).load_state_dict(state)
        if 'best_value' in state:
            self._best_value = state['best_value']


class SaveOnEveryNEpochs(Callback):
    def __init__(self, num_save_epochs: int, *, manager: SessionManager, logger):
        self._num_save_epochs = num_save_epochs
        self._manager = manager
        self._logger = logger

    def after_epoch(self, engine: Trainer, data_loader):
        if distributed.is_local_master() and engine.global_step % self._num_save_epochs == 0:
            self._manager.save(engine.state_dict(), global_step=engine.global_step, logger=self._logger)
