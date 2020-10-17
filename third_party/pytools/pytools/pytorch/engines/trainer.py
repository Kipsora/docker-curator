import abc
import logging
from typing import Optional, Dict, Any, Callable

import torch.utils.data

from pytools.pytorch.engines.engine import Engine
from pytools.pytorch.helpers import load_model_state
from pytools.pytorch.typing import Callback, Optimizer, Scheduler

__all__ = ['Trainer', 'CriterionTrainer']


class Trainer(Engine, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *,
            num_epochs: int,
            callback: Optional[Callback] = None):
        super().__init__(callback)
        self._num_epochs = num_epochs
        self._callback = callback

        self._global_step = -1

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def global_step(self):
        return self._global_step

    def state_dict(self) -> Dict[str, Any]:
        state = super(Trainer, self).state_dict()
        state.update({
            'global_step': self._global_step
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        super(Trainer, self).load_state_dict(state)
        if 'global_step' in state:
            self._global_step = state['global_step']

    @abc.abstractmethod
    def _run_batch(self, inputs):
        raise NotImplementedError()

    def _run_epoch(self, data_loader: torch.utils.data.DataLoader):
        for inputs in data_loader:
            self._prior_batch(inputs)
            outputs = self._run_batch(inputs)
            self._after_batch(inputs, outputs)

    def run(self, data_loader: torch.utils.data.DataLoader):
        self._prior_all()
        while self._global_step < self._num_epochs:
            self._global_step += 1

            self._prior_epoch(data_loader)
            self._run_epoch(data_loader)
            self._after_epoch(data_loader)

        self._after_all()


class CriterionTrainer(Trainer):
    def __init__(
            self,
            model: torch.nn.Module,
            objective: Callable,
            optimizer: Optimizer,
            benchmark: Optional[Callable] = None,
            scheduler: Optional[Scheduler] = None,
            *,
            device: torch.device,
            num_epochs: int,
            callback: Optional[Callback] = None,
            logger: logging.Logger):
        super().__init__(num_epochs=num_epochs, callback=callback)

        self._device = device
        self._model = model
        self._objective = objective
        self._benchmark = benchmark
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._logger = logger

    def state_dict(self) -> Dict[str, Any]:
        state = super(CriterionTrainer, self).state_dict()
        state.update({
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict()
        })
        if self._scheduler is not None:
            state.update({
                'scheduler': self._scheduler.state_dict()
            })
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        super(CriterionTrainer, self).load_state_dict(state)
        if 'model' in state:
            load_model_state(self._model, state['model'], self._logger)
        if 'optimizer' in state:
            self._optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state and self._scheduler is not None:
            self._scheduler.load_state_dict(state['scheduler'])

    def _prior_epoch(self, data_loader):
        super(CriterionTrainer, self)._prior_epoch(data_loader)
        self._model.train()

    def _after_epoch(self, data_loader):
        if self._scheduler is not None:
            self._scheduler.step()
        super(CriterionTrainer, self)._after_epoch(data_loader)

    def _run_batch(self, inputs):
        sources = inputs['source'].to(self._device, non_blocking=True)
        targets = inputs['target'].to(self._device, non_blocking=True)

        self._prior_forward()
        outputs = self._model(sources)
        self._after_forward()

        loss = self._objective(outputs, targets)

        self._optimizer.zero_grad()

        self._prior_backward()
        loss.backward()
        self._optimizer.step()
        self._after_backward()

        result = {'output': outputs, 'objective': loss}
        if self._benchmark is not None:
            result.setdefault('benchmark', self._benchmark(outputs, targets))

        return result
