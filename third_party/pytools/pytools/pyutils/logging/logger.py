import contextlib
import logging
import sys
from typing import List, Type

from pytools.pyutils.logging.formatter import ColoredFormatter, DefaultLogFormat
from pytools.pyutils.logging.group import LogGroup, global_group
from pytools.pyutils.logging.handler import make_handler
from pytools.pyutils.misc.inspect import get_caller_module
from pytools.pyutils.io.file_system import get_relative_path

__all__ = ['get_logger', 'get_ancestor_loggers', 'get_default_logger', 'get_module_logger', 'log_exception_context',
           'get_logger_group']


def get_ancestor_loggers(instance: str or logging.Logger, include_self=False) -> 'List[logging.Logger]':
    if isinstance(instance, str):
        instance = logging.getLogger(instance)
    ancestors = []
    if include_self:
        ancestors.append(instance)
    while instance.parent and instance.propagate:
        instance = instance.parent
        ancestors.append(instance)
    return ancestors


def get_logger_group(logger) -> LogGroup:
    if hasattr(logger, "__group__"):
        return getattr(logger, "__group__")
    else:
        raise RuntimeError("Cannot get the group of an unmanaged logger")


def get_logger(name, level='INFO', group_logger_key: str = None, group: LogGroup = global_group) -> logging.Logger:
    logger = group.attach_logger(name, key=group_logger_key, ignore_reattach=True)
    logger.setLevel(level)
    setattr(logger, "__group__", group)
    return logger


def get_default_logger(name=None,
                       logger_level="NOTSET",
                       group_logger_key: str = None,
                       group_handler_key='__default_console__',
                       group: LogGroup = global_group,
                       subscribe=True,
                       format_kwargs=None,
                       formatter_class: Type[logging.Formatter] = ColoredFormatter,
                       formatter_kwargs=None,
                       handler_level="NOTSET",
                       handler_class=logging.StreamHandler,
                       handler_kwargs=None,
                       log_format=DefaultLogFormat) -> logging.Logger:
    if name is None:
        name = get_relative_path(get_caller_module().__file__)
    group_logger_key = group_logger_key or name
    with group.lock:
        logger = get_logger(name, logger_level, group_logger_key, group)
        handlers = group.lookup_handlers(group_handler_key)
        if not handlers:
            if issubclass(handler_class, logging.StreamHandler):
                default_handler_kwargs = {'stream': sys.stdout}
            else:
                default_handler_kwargs = dict()
            handler = make_handler(
                handler_class,
                level=handler_level,
                format_kwargs=format_kwargs,
                formatter_class=formatter_class,
                formatter_kwargs=formatter_kwargs,
                handler_kwargs=handler_kwargs or default_handler_kwargs,
                log_format=log_format
            )
            group.bind_handler(group_handler_key, handler)
        group.add_loggers_handlers(group_logger_key, group_handler_key)
        if subscribe:
            group.subscribe_logger_handler(group_logger_key, group_handler_key)
        return logger


def get_module_logger(level='INFO',
                      group_logger_key: str = None,
                      group_handler_key='__default_console__',
                      group: LogGroup = global_group,
                      subscribe=True,
                      format_kwargs=None,
                      formatter_class: Type[logging.Formatter] = ColoredFormatter,
                      formatter_kwargs=None,
                      handler_level="NOTSET",
                      handler_class=logging.StreamHandler,
                      handler_kwargs=None):
    module_name = get_caller_module().__name__
    return get_default_logger(module_name, level, group_logger_key, group_handler_key,
                              group, subscribe, format_kwargs, formatter_class,
                              formatter_kwargs, handler_level, handler_class, handler_kwargs)


@contextlib.contextmanager
def log_exception_context(logger: logging.Logger, heading=None):
    try:
        yield
    except SystemExit:
        pass
    except BaseException:
        logger.exception(heading or 'An error has occurred')
        raise
