import collections
import time
from typing import Dict, Any, Union

import numpy as np

from pytools.pytorch.engines import Trainer, CriterionTrainer, CriterionInferer
from pytools.pytorch.engines.callbacks import Callback
from pytools.pytorch.summary import Summary, SummaryReduceMode, SummaryItemType
from pytools.pytorch.typing import Optimizer, Engine

__all__ = ['RecordEstimatedToArrival', 'RecordBatchOutputs', 'RecordLearningRate']


class RecordEstimatedToArrival(Callback):
    def __init__(self, summary: Summary, *, num_epochs_for_average: int = 5, use_scalar_value=True):
        self._num_epochs_for_average = num_epochs_for_average
        self._history_times = None
        self._time_stamp = None
        self._summary = summary
        self._use_scalar_value = use_scalar_value

    def prior_all(self, engine):
        self._history_times = collections.deque(maxlen=self._num_epochs_for_average)
        self._time_stamp = time.time()

    def after_epoch(self, engine: Trainer, data_loader):
        self._history_times.append(time.time() - self._time_stamp)
        self._time_stamp = time.time()

        estimated_time = (engine.num_epochs - engine.global_step) * np.mean(self._history_times)
        self._summary.add_value(
            "eta", estimated_time,
            global_step=engine.global_step,
            item_type=SummaryItemType.ITEM_SCALAR if self._use_scalar_value else SummaryItemType.ITEM_TEMPORARY
        )

    def after_all(self, engine):
        self._history_times = None
        self._time_stamp = None


class RecordBatchOutputs(Callback):
    def __init__(self, summary: Summary, reduction_map: Dict[str, Any], *, prefix="", use_tail_matching=False):
        self._summary = summary
        self._prefix = prefix + "/" if prefix else ""
        self._reduction_map = self._fix_reduction_map(reduction_map)
        self._use_tail_matching = use_tail_matching

    @classmethod
    def _fix_reduction_map(cls, reduction_map: Dict[str, Any]):
        result = dict()
        for key, value in reduction_map.items():
            if isinstance(value, SummaryReduceMode):
                result.setdefault(key, value)
            elif isinstance(value, dict):
                result.setdefault(key, cls._fix_reduction_map(value))
            else:
                result.setdefault(key, SummaryReduceMode.from_string(value))
        return result

    def _record(self, outputs, reduction_map, batch_size, *, global_step, path="", has_matched=False):
        for key, value in outputs.items():
            if key in reduction_map:
                if isinstance(value, dict):
                    self._record(
                        value, reduction_map[key], batch_size,
                        global_step=global_step,
                        path=path + key + "/",
                        has_matched=True)
                else:
                    self._summary.add_value(
                        path + key, value, batch_size,
                        global_step=global_step,
                        reduction=reduction_map[key],
                        item_type=SummaryItemType.ITEM_SCALAR
                    )
            elif not has_matched and self._use_tail_matching and isinstance(value, dict):
                self._record(
                    value, reduction_map, batch_size,
                    global_step=global_step,
                    path=path + key + "/",
                    has_matched=False)

    def after_batch(
            self, engine: Union[CriterionTrainer, CriterionInferer],
            inputs: Dict[str, Any], outputs: Dict[str, Any]):
        self._record(
            outputs, self._reduction_map, len(inputs['source']),
            global_step=engine.global_step,
            path=self._prefix)


class RecordLearningRate(Callback):
    def __init__(self, summary: Summary, optimizer: Optimizer):
        self._optimizer = optimizer
        self._summary = summary

    def prior_epoch(self, engine: Engine, data_loader):
        learning_rates = [group['lr'] for group in self._optimizer.param_groups]
        if len(learning_rates) > 1:
            for index, learning_rate in enumerate(learning_rates):
                self._summary.add_value(
                    f'lr/group{index}', learning_rate,
                    global_step=engine.global_step,
                    item_type=SummaryItemType.ITEM_SCALAR
                )
        else:
            self._summary.add_value(
                f'lr', learning_rates[0],
                global_step=engine.global_step,
                item_type=SummaryItemType.ITEM_SCALAR
            )
