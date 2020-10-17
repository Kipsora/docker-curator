import functools

from pytools.pyutils.io.file_system import get_extname
from pytools.pyutils.misc.registry import RegistryGroup, CallbackRegistry

__all__ = ['io_registry_group']


class CallbackRegistryGroup(RegistryGroup):
    __base_class__ = CallbackRegistry

    def _default_io_fallback(self, extension, *args, **kwargs):
        raise ValueError(f"Unknown file extension: {extension}")

    def lookup(self, registry_name, key, *args, **kwargs):
        extension = get_extname(key)
        callback = super(CallbackRegistryGroup, self).lookup(registry_name, extension, True, self._default_io_fallback)
        return callback(key, *args, **kwargs)

    def __getattr__(self, item):
        return functools.partial(self.lookup, item)


io_registry_group = CallbackRegistryGroup()
