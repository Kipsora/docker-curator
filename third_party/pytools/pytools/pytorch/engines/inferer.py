import abc
from typing import Optional

import torch.utils.data

from pytools.pytorch.engines.engine import Engine
from pytools.pytorch.typing import Callback

__all__ = ['Inferer', 'CriterionInferer']


class Inferer(Engine):
    def __init__(self, callback: Optional[Callback]):
        super().__init__(callback)

        self._global_step = None
        self._name = None

    @property
    def global_step(self):
        if self._global_step is None:
            raise ValueError("The global_step has not been set by the inferer")
        return self._global_step

    def run(self, data_loader: torch.utils.data.DataLoader, *, dataset_name=None, global_step=None):
        self._global_step = global_step

        self._prior_all()
        self._prior_epoch(data_loader)
        for inputs in data_loader:
            with torch.no_grad():
                self._prior_batch(inputs)
                outputs = self._run_batch(inputs)
                if dataset_name:
                    outputs = {dataset_name: outputs}
                self._after_batch(inputs, outputs)
        self._after_epoch(data_loader)
        self._after_all()

    @abc.abstractmethod
    def _run_batch(self, inputs):
        raise NotImplementedError


class CriterionInferer(Inferer):
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: Optional[torch.nn.Module] = None,
            *,
            device: torch.device,
            callback: Optional[Callback] = None):
        super().__init__(callback)
        self._device = device
        self._model = model
        self._criterion = criterion

    def _prior_epoch(self, data_loader: torch.utils.data.DataLoader):
        super(CriterionInferer, self)._prior_epoch(data_loader)
        self._model.eval()

    def _run_batch(self, inputs):
        sources = inputs['source'].to(self._device, non_blocking=True)
        targets = inputs['target'].to(self._device, non_blocking=True) if 'target' in inputs else None

        self._prior_forward()
        outputs = self._model(sources)
        self._after_forward()

        results = {'output': outputs}

        if targets is not None and self._criterion is not None:
            metrics = self._criterion(outputs, targets)
            results.update({'criterion': metrics})

        return results
