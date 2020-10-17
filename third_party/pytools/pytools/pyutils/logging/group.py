import contextlib
import logging
import threading
from typing import Set

from pytools.pyutils.misc.decorator import synchronized_member_fn
from pytools.pyutils.misc.relation import BiRelation

__all__ = ['LogGroup', 'global_group']


class LogGroup(object):
    def __init__(self):
        self._lock = threading.RLock()

        self._managed_loggers = BiRelation()
        self._managed_filters = BiRelation()
        self._managed_handlers = BiRelation()

        self._logger_filter_subs = BiRelation()
        self._logger_handler_subs = BiRelation()
        self._handler_filter_subs = BiRelation()

    @property
    def lock(self):
        return self._lock

    @synchronized_member_fn
    def subscribe_logger_handler(self, logger_key, handler_key):
        return self._logger_handler_subs.set(logger_key, handler_key)

    @synchronized_member_fn
    def unsubscribe_logger_handler(self, logger_key, handler_key):
        return self._logger_handler_subs.unset(logger_key, handler_key)

    @synchronized_member_fn
    def subscribe_logger_filter(self, logger_key, filter_key):
        return self._logger_filter_subs.set(logger_key, filter_key)

    @synchronized_member_fn
    def unsubscribe_logger_filter(self, logger_key, filter_key):
        return self._logger_filter_subs.unset(logger_key, filter_key)

    @synchronized_member_fn
    def subscribe_handler_filter(self, handler_key, filter_key):
        return self._handler_filter_subs.set(handler_key, filter_key)

    @synchronized_member_fn
    def unsubscribe_handler_filter(self, handler_key, filter_key):
        return self._handler_filter_subs.unset(handler_key, filter_key)

    @synchronized_member_fn
    def has_handler_key(self, key: str):
        return self._managed_handlers.has_source(key)

    @synchronized_member_fn
    def has_handler(self, instance: logging.Handler):
        return self._managed_handlers.has_target(instance)

    @synchronized_member_fn
    def has_filter_key(self, key: str):
        return self._managed_filters.has_source(key)

    @synchronized_member_fn
    def has_filter(self, instance: logging.Filter):
        return self._managed_filters.has_target(instance)

    @synchronized_member_fn
    def has_logger(self, key: str):
        return self._managed_loggers.has_source(key)

    @synchronized_member_fn
    def has_logger_key(self, instance: logging.Logger):
        return self._managed_loggers.has_target(instance)

    @synchronized_member_fn
    def bind_handler(self, key: str, instance: logging.Handler, ignore_rebind=False):
        assert key is None or isinstance(key, str)
        if not self._managed_handlers.set(key, instance):
            if not ignore_rebind:
                raise ValueError(f'The handler pair "({key}, {instance})" has already been registered in this group')
        else:
            for logger in self.as_loggers(self._logger_handler_subs.get_sources(key)):
                logger.addHandler(instance)
        return instance

    @synchronized_member_fn
    def unbind_handler(self, key: str, instance: logging.Handler = None):
        assert key is None or isinstance(key, str)
        if self._managed_handlers.unset(key, instance):
            for logger in self.as_loggers(self._logger_handler_subs.get_sources(key)):
                logger.removeHandler(instance)
            return True
        return False

    @synchronized_member_fn
    def lookup_handlers(self, key: str = None) -> 'Set[logging.Handler]':
        assert key is None or isinstance(key, str)
        return self._managed_handlers.get_targets(key)

    @synchronized_member_fn
    def lookup_handler_keys(self, instance: logging.Handler = None):
        return self._managed_handlers.get_sources(instance)

    @synchronized_member_fn
    def as_handlers(self, instances=None) -> 'Set[logging.Handler]':
        instances = {instances} if isinstance(instances, (type(None), str, logging.Handler)) else set(instances)
        return set.union(*[self.lookup_handlers(instance)
                           if instance is None or isinstance(instance, str) else {instance}
                           for instance in instances]) if instances else {}

    @synchronized_member_fn
    def as_handler_keys(self, instances=None) -> 'Set[str]':
        instances = {instances} if isinstance(instances, (type(None), str, logging.Handler)) else set(instances)
        return set.union(*[self.lookup_handler_keys(instance)
                           if instance is None or isinstance(instance, logging.Handler) else {instance}
                           for instance in instances]) if instances else {}

    @synchronized_member_fn
    def bind_filter(self, key: str, instance: logging.Filter = None, ignore_rebind=False):
        assert key is None or isinstance(key, str)
        if not self._managed_filters.set(key, instance):
            if not ignore_rebind:
                raise ValueError(f'The filter pair "({key}, {instance})" has already been registered in this group')
        else:
            for logger in self.as_loggers(self._logger_filter_subs.get_sources(key)):
                logger.addFilter(instance)
            for handler in self.as_handlers(self._handler_filter_subs.get_sources(key)):
                handler.addFilter(instance)
        return instance

    @synchronized_member_fn
    def unbind_filter(self, key: str, instance: logging.Filter = None):
        assert key is None or isinstance(key, str)
        if self._managed_handlers.unset(key, instance):
            for logger in self.as_loggers(self._logger_filter_subs.get_sources(key)):
                logger.removeFilter(instance)
            for handler in self.as_handlers(self._handler_filter_subs.get_sources(key)):
                handler.removeFilter(instance)
            return True
        return False

    @synchronized_member_fn
    def lookup_filters(self, key: str = None) -> 'Set[logging.Filter]':
        assert key is None or isinstance(key, str)
        return self._managed_filters.get_targets(key)

    @synchronized_member_fn
    def lookup_filter_keys(self, instance: logging.Filter = None) -> 'Set[str]':
        return self._managed_filters.get_targets(instance)

    @synchronized_member_fn
    def as_filters(self, instances=None) -> 'Set[logging.Filter]':
        instances = {instances} if isinstance(instances, (type(None), str, logging.Filter)) else set(instances)
        return set.union(*[self.lookup_filters(instance)
                           if instance is None or isinstance(instance, str) else {instance}
                           for instance in instances]) if instances else {}

    @synchronized_member_fn
    def as_filter_keys(self, instances=None) -> 'Set[str]':
        instances = {instances} if isinstance(instances, (type(None), str, logging.Filter)) else set(instances)
        return set.union(*[self.lookup_filter_keys(instance)
                           if instance is None or isinstance(instance, logging.Filter) else {instances}
                           for instance in set(instances)]) if instances else {}

    @synchronized_member_fn
    def attach_logger(self, instance: str or logging.Logger, key=None, ignore_reattach=False):
        if isinstance(instance, str):
            instance = logging.getLogger(instance)
        if not self._managed_loggers.set(key or instance.name, instance) and not ignore_reattach:
            raise KeyError(f'The logger pair "({key}, {instance})" has already been registered in this group')
        return instance

    @synchronized_member_fn
    def detach_logger(self, instance: str or logging.Logger):
        if isinstance(instance, str):
            instance = logging.getLogger(instance)
        self._managed_loggers.unset(target=instance)
        return instance

    @synchronized_member_fn
    def lookup_loggers(self, key=None) -> 'Set[logging.Logger]':
        return self._managed_loggers.get_targets(key)

    @synchronized_member_fn
    def lookup_logger_keys(self, instance: str or logging.Logger) -> 'Set[str]':
        if isinstance(instance, str):
            instance = logging.getLogger(instance)
        return self._managed_loggers.get_sources(instance)

    @synchronized_member_fn
    def as_loggers(self, instances=None) -> 'Set[logging.Logger]':
        instances = {instances} if isinstance(instances, (type(None), str, logging.Logger)) else set(instances)
        return set.union(*[self.lookup_loggers(instance)
                           if instance is None or isinstance(instance, str) else {instance}
                           for instance in instances]) if instances else {}

    @synchronized_member_fn
    def as_logger_keys(self, instances=None) -> 'Set[str]':
        instances = {instances} if isinstance(instances, (type(None), str, logging.Logger)) else set(instances)
        return set.union(*[self.lookup_logger_keys(instance)
                           if instance is None or isinstance(instance, logging.Logger) else {instance}
                           for instance in instances]) if instances else {}

    def set_logger_level(self, level, loggers=None):
        if isinstance(level, str):
            level = level.upper()
        for logger in self.as_loggers(loggers):
            logger.setLevel(level)

    def set_handler_level(self, level, handlers=None):
        if isinstance(level, str):
            level = level.upper()
        for instance in self.as_handlers(handlers):
            instance.setLevel(level)

    def add_loggers_handlers(self, loggers=None, handlers=None):
        for logger in self.as_loggers(loggers):
            for instance in self.as_handlers(handlers):
                logger.addHandler(instance)

    def del_loggers_handlers(self, loggers=None, handlers=None):
        for logger in self.as_loggers(loggers):
            for instance in self.as_handlers(handlers):
                logger.removeHandler(instance)

    def add_loggers_filters(self, loggers=None, filters=None):
        for logger in self.as_loggers(loggers):
            for instance in self.as_filters(filters):
                logger.addFilter(instance)

    def del_loggers_filters(self, loggers=None, filters=None):
        for logger in self.as_loggers(loggers):
            for instance in self.as_filters(filters):
                logger.removeFilter(instance)

    def set_handler_formatter(self, handlers=None, *, formatter: logging.Formatter):
        for instance in self.as_handlers(handlers):
            instance.setFormatter(formatter)

    def add_handler_filters(self, handlers=None, filters=None):
        for handler in self.as_handlers(handlers):
            for instance in self.as_filters(filters):
                handler.addFilter(instance)

    def del_handler_filters(self, handlers=None, filters=None):
        for handler in self.as_handlers(handlers):
            for instance in self.as_filters(filters):
                handler.removeFilter(instance)

    @contextlib.contextmanager
    def switch_handler_context(self, source_handlers, target_handlers, loggers=None,
                               use_same_level=True, use_same_formatter=True):
        source_handlers = self.as_handlers(source_handlers)
        target_handlers = self.as_handlers(target_handlers)
        add_handlers = target_handlers.difference(source_handlers)
        del_handlers = source_handlers.difference(target_handlers)
        restore_add_handlers = []
        restore_del_handlers = []
        origin_target_levels = [handler.level for handler in target_handlers] if use_same_level else []
        origin_target_formatters = [handler.formatter for handler in target_handlers] if use_same_formatter else []
        for logger in self.as_loggers(loggers):
            for instance in del_handlers:
                logger.removeHandler(instance)
                restore_add_handlers.append((logger, instance))
            for instance in add_handlers:
                logger.addHandler(instance)
                restore_del_handlers.append((logger, instance))
        try:
            yield
        finally:
            if use_same_level:
                for handler, level in zip(target_handlers, origin_target_levels):
                    handler.setLevel(level)
            if use_same_formatter:
                for handler, formatter in zip(target_handlers, origin_target_formatters):
                    handler.setFormatter(formatter)
            for logger, instance in restore_add_handlers:
                logger.addHandler(instance)
            for logger, instance in restore_del_handlers:
                logger.removeHandler(instance)

    @contextlib.contextmanager
    def switch_filter_context(self, source_filters, target_filters, loggers=None):
        source_filters = self.as_filters(source_filters)
        target_filters = self.as_filters(target_filters)
        add_filters = target_filters.difference(source_filters)
        del_filters = source_filters.difference(target_filters)
        restore_add_filters = []
        restore_del_filters = []
        for logger in self.as_loggers(loggers):
            for instance in del_filters:
                logger.removeFilter(instance)
                restore_add_filters.append((logger, instance))
            for instance in add_filters:
                logger.addFilter(instance)
                restore_del_filters.append((logger, instance))
        try:
            yield
        finally:
            for logger, instance in restore_add_filters:
                logger.addFilter(instance)
            for logger, instance in restore_del_filters:
                logger.removeFilter(instance)

    @contextlib.contextmanager
    def handler_level_context(self, level, handlers=None):
        if isinstance(level, str):
            level = level.upper()
        handlers = self.as_handlers(handlers)
        restore_set_levels = []
        for instance in handlers:
            restore_set_levels.append((instance, instance.level))
            instance.setLevel(level)
        try:
            yield
        finally:
            for instance, level in restore_set_levels:
                instance.setLevel(level)

    @contextlib.contextmanager
    def logger_level_context(self, level, loggers=None):
        if isinstance(level, str):
            level = level.upper()
        loggers = self.as_loggers(loggers)
        restore_set_levels = []
        for logger in loggers:
            restore_set_levels.append((logger, logger.level))
            logger.setLevel(level)
        try:
            yield
        finally:
            for logger, level in restore_set_levels:
                logger.setLevel(level)


global_group = LogGroup()


def __getattr__(name):
    return getattr(global_group, name)
