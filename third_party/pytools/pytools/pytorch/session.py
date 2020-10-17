import contextlib
import datetime
import logging
import os
import re
from typing import Optional

import filelock
import torch

from pytools.pytorch import distributed
from pytools.pyutils.io.file_system import ensure_directory, remove, get_absolute_path, get_relative_path
from pytools.pyutils.io.pretty import load_json, dump_json
from pytools.pyutils.logging.handler import make_handler
from pytools.pyutils.logging.logger import get_default_logger, get_logger_group
from pytools.pyutils.misc.decorator import cached_property
from pytools.pyutils.misc.time_stamp import current_time

__all__ = ['SessionManager']


class SessionManager(object):
    __datefmt__ = '%Y-%m-%d %H:%M:%S'
    __default__: Optional['SessionManager'] = None

    @classmethod
    def default_manager(cls):
        if cls.__default__ is None:
            raise ValueError("The default session manager has not been set")
        return cls.__default__

    @contextlib.contextmanager
    def as_default(self):
        original_default_manager = self.__class__.__default__
        try:
            self.__class__.__default__ = self
            yield
        finally:
            self.__class__.__default__ = original_default_manager

    def __init__(
            self,
            session_path: str,
            session_type: str,
            session_time: str,
            num_sessions: int = 20,
            restore_global_step: Optional[int] = None
    ):
        self._session_path = session_path
        self._session_type = session_type
        self._session_time = session_time

        self._num_sessions = num_sessions
        self._restore_global_step = restore_global_step

    @property
    def session_type(self):
        return self._session_type

    @property
    def session_path(self):
        return self._session_path

    @property
    def session_time(self):
        return self._session_time

    @property
    def restore_global_step(self):
        return self._restore_global_step

    @cached_property
    def current_session_path(self):
        path = os.path.join(self.session_path, self.session_type, self.session_time)
        return path

    @classmethod
    def _load_checkpoint_logs(cls, session_path):
        if os.path.exists(os.path.join(session_path, "checkpoints.json")):
            return load_json(os.path.join(session_path, "checkpoints.json"))
        else:
            return {'checkpoints': []}

    @classmethod
    def _save_checkpoint_logs(cls, session_path, logs):
        dump_json(os.path.join(session_path, "checkpoints.json"), logs)

    @classmethod
    def _get_latest_epoch(cls, path):
        engine = re.compile(r"model_(\d*).pth")
        model_names = sorted([
            name for name in os.listdir(path)
            if os.path.isfile(os.path.join(path, name)) and engine.match(name)
        ])
        if not model_names:
            return None
        model_name = model_names[-1]

        global_step, = re.findall(r"model_(\d*).pth", model_name)
        global_step = int(global_step)

        return global_step

    @classmethod
    def _get_latest_time(cls, path, *, with_global_step=False):
        for name in os.listdir(path):
            try:
                session_time = datetime.datetime.strptime(name, cls.__datefmt__)
                session_time = session_time.strftime(cls.__datefmt__)
                global_step = cls._get_latest_epoch(os.path.join(path, session_time))
                if global_step is None:
                    continue
                return session_time if not with_global_step else (session_time, global_step)
            except ValueError:
                continue
        return None

    def get_latest_epoch(self):
        global_step = self._get_latest_epoch(self.current_session_path)
        if global_step is None:
            raise RuntimeError(f"Cannot find any valid model in path {self.current_session_path}")
        return global_step

    def load(self, *, global_step=None, logger: logging.Logger):
        if global_step is None:
            if self._restore_global_step is not None:
                global_step = self._restore_global_step
                self._restore_global_step = None
            else:
                global_step = self.get_latest_epoch()

        checkpoint = os.path.join(self.current_session_path, f"model_{global_step}.pth")

        logger.info(f"==> Loading from checkpoint {checkpoint}...")

        if not os.path.isfile(checkpoint):
            raise RuntimeError(f"Checkpoint file model_{global_step}.pth does not exist")

        with filelock.FileLock(os.path.join(self.current_session_path, f"checkpoints.json.lock")):
            checkpoint_logs = self._load_checkpoint_logs(self.current_session_path)
            if f"model_{global_step}.pth" not in checkpoint_logs['checkpoints']:
                raise RuntimeError(f"The logs has not record checkpoint model_{global_step}.pth")

            return torch.load(checkpoint)

    @classmethod
    def _get_unified_time(cls):
        return distributed.broadcast(current_time(cls.__datefmt__))

    @classmethod
    def from_session_type(cls, session_path, session_type, *, num_sessions: int = 20):
        return SessionManager(session_path, session_type, cls._get_unified_time(), num_sessions=num_sessions)

    @classmethod
    def from_restore_path(cls, path, *, num_sessions: int = 20):
        path = get_absolute_path(path)

        if not os.path.exists(path):
            raise RuntimeError(f"The path to be loaded does not exists")

        try:
            session_time = None
            if os.path.isfile(path):
                model_name = os.path.basename(path)
                path = os.path.dirname(path)

                global_step, = re.findall(r"model_(\d*).pth", model_name)
                global_step = int(global_step)
            else:
                global_step = cls._get_latest_epoch(path)
                if global_step is None:
                    result = cls._get_latest_time(path, with_global_step=True)
                    if result is None:
                        raise ValueError(f"Cannot find any valid session in the path {path}")
                    session_time, global_step = result
                    path = os.path.join(path, session_time)

            if session_time is not None:
                time_stamp = datetime.datetime.strptime(os.path.basename(path), cls.__datefmt__)
                session_time = time_stamp.strftime(cls.__datefmt__)
                path = os.path.dirname(path)

            session_type = os.path.basename(path)
            path = os.path.dirname(path)

            session_path = get_absolute_path(path, use_real_path=True)

            return SessionManager(
                session_path, session_type, session_time,
                num_sessions=num_sessions,
                restore_global_step=global_step
            )
        except ValueError:
            raise RuntimeError("The path is not in the form of <session_path>/<session_type>"
                               "[/<time>][/model_<global_step>.pth]")

    def save(self, data, *, global_step, logger: logging.Logger):
        checkpoint = os.path.join(self.current_session_path, f"model_{global_step}.pth")
        logger.info(f"==> Saving session of global step {global_step} to {checkpoint}...")

        with filelock.FileLock(os.path.join(self.current_session_path, f"checkpoints.json.lock")):
            checkpoint_logs = self._load_checkpoint_logs(self.current_session_path)

            while len(checkpoint_logs['checkpoints']) >= self._num_sessions > 0:
                removed_checkpoint = checkpoint_logs['checkpoints'].pop(0)
                remove(os.path.join(self.current_session_path, removed_checkpoint))
                logger.debug(f"Removed checkpoint {removed_checkpoint}")

            torch.save(data, checkpoint)
            checkpoint_logs['checkpoints'].append(get_relative_path(checkpoint, self.current_session_path))

            self._save_checkpoint_logs(self.current_session_path, checkpoint_logs)

    def get_log_path(self, name):
        log_time = self._get_unified_time()
        if distributed.is_distribute_enabled() and distributed.get_size() > 1:
            return os.path.join(self.current_session_path, f"{name}_{distributed.get_rank()}_{log_time}.log")
        return os.path.join(self.current_session_path, f"{name}_{log_time}.log")

    def get_tboard_path(self):
        return os.path.join(self.current_session_path, "tensorboard")

    def get_logger(self, name, *, is_verbose, use_log_file=True):
        logger = get_default_logger(name, "DEBUG", handler_level="DEBUG" if is_verbose else "INFO")
        if distributed.is_distribute_enabled() and distributed.get_local_rank() != 0:
            get_logger_group(logger).del_loggers_handlers(logger, "__default_console__")

        if use_log_file:
            file_handler = make_handler(logging.FileHandler, "DEBUG", handler_kwargs={
                "filename": self.get_log_path(name)
            })
            get_logger_group(logger).add_loggers_handlers(logger, file_handler)
        return logger

    def make_session_path(self):
        if distributed.is_local_master():
            ensure_directory(self.current_session_path)
        distributed.barrier()
