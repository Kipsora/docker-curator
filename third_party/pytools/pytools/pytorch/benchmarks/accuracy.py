import torch

__all__ = ['TopKAccuracy']


class TopKAccuracy(torch.nn.Module):
    def __init__(self, top_ranks=(1,)):
        super().__init__()
        self._top_ranks = top_ranks

    @property
    def reduction(self):
        return {f'top_{k}': 'global_mean' for k in self._top_ranks}

    @torch.no_grad()
    def forward(self, outputs, targets):
        max_rank = max(self._top_ranks)

        _, pred = outputs.topk(max_rank, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        result = {}
        for k in self._top_ranks:
            correct_k = correct[:k].view(-1).float().sum(0)
            result.setdefault(f'top_{k}', correct_k)

        return result
