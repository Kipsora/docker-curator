import abc

import torch

from pytools.legacy.pytorch.criterion.base import Criterion

__all__ = ['Benchmark']


class Benchmark(Criterion, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @torch.no_grad()
    def forward(self, outputs, targets):
        raise NotImplementedError()
