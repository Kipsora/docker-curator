import collections
import enum
from typing import Optional, Dict, Deque, Union

import numpy as np
import torch

from pytools.pytorch import distributed

__all__ = ['Summary', 'SummaryReduceMode', 'SummaryItemType']

SummaryNumber = Union[int, float, np.ndarray, np.floating, np.integer, torch.Tensor]


class SummaryItem(object):
    def __init__(self, value: SummaryNumber, count: Optional[SummaryNumber]):
        self.value = value
        self.count = count


class SummaryReduceMode(enum.Enum):
    REDUCE_SUM = enum.auto()
    REDUCE_MEAN = enum.auto()
    REDUCE_GLOBAL_MEAN = enum.auto()

    @classmethod
    def is_reduce_mean(cls, mode: Optional['SummaryReduceMode']):
        return mode == SummaryReduceMode.REDUCE_MEAN or mode == SummaryReduceMode.REDUCE_GLOBAL_MEAN

    @classmethod
    def to_string(cls, mode: Optional['SummaryReduceMode']):
        if mode is None:
            return 'none'
        elif mode == SummaryReduceMode.REDUCE_SUM:
            return 'sum'
        elif mode == SummaryReduceMode.REDUCE_MEAN:
            return 'mean'
        elif mode == SummaryReduceMode.REDUCE_GLOBAL_MEAN:
            return 'global_mean'
        else:
            raise ValueError(f"Unexpected mode {mode}")

    @classmethod
    def from_string(cls, mode: Optional[str]):
        if mode is None or mode == 'none':
            return None
        elif mode == 'sum':
            return SummaryReduceMode.REDUCE_SUM
        elif mode == 'mean':
            return SummaryReduceMode.REDUCE_MEAN
        elif mode == 'global_mean':
            return SummaryReduceMode.REDUCE_GLOBAL_MEAN
        else:
            raise ValueError(f"Unexpected mode string {mode}")


class SummaryItemType(enum.Enum):
    ITEM_SCALAR = enum.auto()
    ITEM_TEMPORARY = enum.auto()


class SummaryStorage(object):
    def __init__(
            self,
            *,
            item_type: SummaryItemType,
            reduction: Optional[SummaryReduceMode],
            num_items_kept: Optional[int]
    ):
        self._storage: Dict[int, SummaryItem] = dict()
        self._indices: Deque[int] = collections.deque()
        self.item_type = item_type
        self.reduction = reduction
        self.num_items_kept = num_items_kept

    @property
    def indices(self):
        return self._indices

    def _copy_number(self, value: Optional[SummaryNumber]):
        if value is None:
            return value

        if SummaryReduceMode.is_reduce_mean(self.reduction):
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach()
                value.requires_grad = False
            else:
                value = torch.tensor(value)
        elif isinstance(value, torch.Tensor):
            value = value.cpu().detach()
            value.requires_grad = False
        elif isinstance(value, np.ndarray):
            value = value.copy()
        return value

    def add_value(
            self,
            value: SummaryNumber,
            count: Optional[SummaryNumber],
            *,
            global_step: Optional[int] = None):
        if global_step is None:
            global_step = self._indices[-1]
        elif self._indices and global_step < self._indices[-1]:
            raise ValueError("Cannot add value to previous epoch")

        value = self._copy_number(value)
        count = self._copy_number(count)

        if self.reduction == SummaryReduceMode.REDUCE_MEAN and count is not None:
            value *= count

        if global_step not in self._storage:
            if SummaryReduceMode.is_reduce_mean(self.reduction) and count is None:
                count = torch.tensor(1)

            self._storage[global_step] = SummaryItem(value, count)
            self._indices.append(global_step)

            while len(self._indices) > self.num_items_kept:
                self._storage.pop(self._indices.popleft())

            return

        item = self._storage[global_step]

        if isinstance(item.value, np.ndarray) and isinstance(value, torch.Tensor):
            value = value.numpy()
        if isinstance(item.count, np.ndarray) and isinstance(value, torch.Tensor):
            count = count.numpy()

        item.value += value
        if item.count is not None:
            item.count += (1 if count is None else count)

    def get_value(self, *, global_step: Optional[int] = None) -> SummaryNumber:
        if global_step is None:
            global_step = self._indices[-1]

        item = self._storage[global_step]

        if not SummaryReduceMode.is_reduce_mean(self.reduction):
            return item.value

        return item.value / item.count

    def synchronize(self, *, global_step: Optional[int] = None, use_async_op: bool = False):
        if self.reduction is None:
            return list() if use_async_op else None

        if global_step is None:
            global_step = self._indices[-1]

        item = self._storage[global_step]
        futures = distributed.all_reduce(item.value, use_async_op=True)
        if item.count is not None:
            futures.extend(distributed.all_reduce(item.count, use_async_op=True))

        if use_async_op:
            return futures

        for future in futures:
            future.wait()

    def __len__(self):
        return len(self._indices)

    def __bool__(self):
        return bool(self._indices)


class Summary(object):
    def __init__(self):
        self._history: Dict[str, SummaryStorage] = dict()

    def _record(
            self,
            name: str,
            value: SummaryNumber,
            count: Optional[SummaryNumber],
            *,
            item_type: SummaryItemType,
            global_step: Optional[int],
            reduction: Optional[SummaryReduceMode] = None,
            num_epochs_kept: Optional[int] = 1):

        if name not in self._history:
            storage = SummaryStorage(
                item_type=item_type,
                reduction=reduction,
                num_items_kept=num_epochs_kept
            )
            self._history.setdefault(name, storage)

        storage = self._history[name]

        if storage.item_type != item_type or storage.reduction != reduction or \
                storage.num_items_kept != num_epochs_kept:
            raise ValueError(f"Different storage settings were found at record {name}")

        storage.add_value(value, count, global_step=global_step)

    def add_value(
            self,
            name: str,
            value: SummaryNumber,
            count: Optional[SummaryNumber] = None,
            *,
            item_type: SummaryItemType,
            global_step: Optional[int] = None,
            reduction: Optional[SummaryReduceMode] = None,
            num_epochs_kept: Optional[int] = 1):
        self._record(
            name, value, count,
            global_step=global_step,
            reduction=reduction,
            num_epochs_kept=num_epochs_kept,
            item_type=item_type)

    def names(self):
        return self._history.keys()

    def storages(self):
        return self._history.values()

    def __getitem__(self, item):
        return self._history[item]

    def __contains__(self, item):
        return item in self._history
