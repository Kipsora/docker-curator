from typing import Any, Dict

import tqdm

from pytools.pytorch import distributed
from pytools.pytorch.engines.callbacks import Callback
from pytools.pytorch.summary import Summary
from pytools.pytorch.typing import Engine
from pytools.pyutils.logging.group import LogGroup, global_group
from pytools.pyutils.logging.handler import CallbackHandler
from pytools.pyutils.misc.string import WildCardMatcher

__all__ = ['ShowEpochProgress']


class ShowEpochProgress(Callback):
    def __init__(
            self,
            summary: Summary,
            matcher: WildCardMatcher = WildCardMatcher("*"),
            *,
            logger_group: LogGroup = global_group,
            num_digits: int = 3,
            console_handler_key: str = "__default_console__"
    ):
        self._bar = None
        self._logger_group = logger_group
        self._num_digits = num_digits
        self._console_handler_key = console_handler_key
        self._context = None

        self._summary = summary
        self._matcher = matcher

    def prior_epoch(self, engine, data_loader):
        if distributed.is_local_master():
            self._bar = tqdm.tqdm(total=len(data_loader), ncols=0, leave=False)
            self._context = self._logger_group.switch_handler_context(
                self._console_handler_key, CallbackHandler(self._bar.write))
            self._context.__enter__()

    def after_batch(self, engine: Engine, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        if distributed.is_local_master():
            texts = []
            for name in sorted(self._summary.names()):
                if self._matcher.match(name) and self._summary[name].indices[-1] == engine.global_step:
                    texts.append(f'[{name}] = {self._summary[name].get_value():.{self._num_digits}g}')
            if texts:
                self._bar.set_description(', '.join(texts))
            self._bar.update()

    def after_epoch(self, engine, data_loader):
        if distributed.is_local_master():
            self._bar.close()
            self._bar = None

            self._context.__exit__(None, None, None)
            self._context = None
