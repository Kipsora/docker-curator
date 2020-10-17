import torch

from pytools.legacy.pytorch import curator
from pytools.legacy.pytorch.criterion.benchmarks import Benchmark

__all__ = ['TopKAccuracy']


@curator.register_module('criterion/benchmark')
class TopKAccuracy(Benchmark):
    def __init__(self, top_ranks=(1,), reduction='sum'):
        super().__init__()
        self._top_ranks = top_ranks
        self._reduction = reduction

    @property
    def reduction(self):
        return {
            f'top_{k}': self._reduction
            for k in self._top_ranks
        }

    @torch.no_grad()
    def forward(self, outputs, targets):
        max_rank = max(self._top_ranks)

        _, pred = outputs.topk(max_rank, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        result = {}
        for k in self._top_ranks:
            if self._reduction == 'sum':
                correct_k = correct[:k].view(-1).float().sum(0)
            else:
                correct_k = correct[:k].view(-1, k).float().sum(-1)
            result.setdefault(f'top_{k}', correct_k)

        return result
