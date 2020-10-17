import logging

__all__ = ['LoggerWrapper']

from pytools.pyutils.logging.logger import get_logger_group


class LoggerWrapper(object):
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def log(self, level, msg, *args, **kwargs):
        msg = f"{msg} " + " ".join(map(repr, args))
        return self._logger.log(level, msg, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self.log(logging.INFO, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return self.log(logging.CRITICAL, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self.log(logging.WARNING, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        return self.log(logging.DEBUG, msg, *args, **kwargs)

    @property
    def group(self):
        return get_logger_group(self._logger)
