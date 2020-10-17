import abc
import typing

import torch

__all__ = ['Criterion', 'ComposeCriterion']


class Criterion(torch.nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def reduction(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, outputs, targets):
        raise NotImplementedError()


class ComposeCriterion(Criterion):
    def __init__(self, criterion: typing.Dict[str, Criterion]):
        super().__init__()
        self._criterion = criterion
        for k, v in self._criterion.items():
            self.add_module(k, v)

    @property
    def reduction(self):
        return {k: v.reduction for k, v in self._criterion.items()}

    def forward(self, outputs, targets):
        return {k: v.forward(outputs, targets) for k, v in self._criterion.items()}
