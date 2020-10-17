import logging
from typing import Optional, Tuple, List

from pytools.pytorch import distributed
from pytools.pytorch.engines.callbacks import Callback
from pytools.pytorch.session import SessionManager
from pytools.pytorch.summary import SummaryReduceMode, SummaryItemType
from pytools.pytorch.typing import Summary, Engine

__all__ = ['SyncSummary', 'WriteSummaryToLogger', 'WriteSummaryToTBoard']

from pytools.pyutils.misc.string import WildCardMatcher

from pytools.pyutils.misc.string.formatters import NumberFormatter, TemplateFormatter


class SyncSummary(Callback):
    def __init__(self, summary: Summary):
        self._summary = summary

    def after_epoch(self, engine: Engine, data_loader):
        futures = list()
        for name in self._summary.names():
            storage = self._summary[name]
            if not storage or storage.indices[-1] != engine.global_step:
                continue

            futures.extend(storage.synchronize(global_step=engine.global_step, use_async_op=True))

        for future in futures:
            future.wait()


class WriteSummaryToLogger(Callback):
    def __init__(self, summary: Summary, logger: logging.Logger, *,
                 formatter_filters: Optional[List[Tuple[WildCardMatcher, NumberFormatter]]] = None):
        self._summary = summary
        self._logger = logger
        self._formatter_filters = formatter_filters or [(WildCardMatcher('*'), TemplateFormatter('.7g'))]

    def _find_formatter(self, name):
        for matcher, formatter in self._formatter_filters:
            if not matcher.match(name):
                continue
            return formatter

    def after_epoch(self, engine: Engine, data_loader):
        data = {
            'Path': ('Reduction', 'Value')
        }

        for name in sorted(self._summary.names()):
            formatter = self._find_formatter(name) or str

            storage = self._summary[name]
            value = formatter(storage.get_value())
            reduction = SummaryReduceMode.to_string(storage.reduction)

            if storage.indices[-1] != engine.global_step:
                data.setdefault(name + "!", (reduction, value))
            else:
                data.setdefault(name, (reduction, value))

        max_name_length = 0
        max_mode_length = 0
        max_value_length = 0
        for key, (reduction, value) in data.items():
            max_name_length = max(max_name_length, len(key))
            max_mode_length = max(max_mode_length, len(reduction))
            max_value_length = max(max_value_length, len(value))

        message = f"==> Epoch {engine.global_step}:\n"
        for key, (reduction, value) in data.items():
            message += f"| {key:{max_name_length}s} | {reduction:{max_mode_length}s} | {value:{max_value_length}s} |\n"

        self._logger.info(message)


class WriteSummaryToTBoard(Callback):
    def __init__(self, summary: Summary, *, path: Optional[str] = None):
        self._path = path or SessionManager.default_manager().get_tboard_path()
        self._writer = None
        self._summary = summary

    def prior_all(self, engine: Engine):
        if distributed.is_local_master():
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self._path, purge_step=engine.global_step)

    def after_epoch(self, engine: Engine, data_loader):
        if distributed.is_local_master():
            for name in sorted(self._summary.names()):
                storage = self._summary[name]
                if not storage or storage.indices[-1] != engine.global_step:
                    continue

                if storage.item_type == SummaryItemType.ITEM_SCALAR:
                    self._writer.add_scalar(
                        name,
                        storage.get_value(global_step=engine.global_step),
                        global_step=engine.global_step
                    )

    def after_all(self, engine: Engine):
        if distributed.is_local_master():
            self._writer.close()
            self._writer = None
