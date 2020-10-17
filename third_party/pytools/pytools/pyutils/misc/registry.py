import collections
import functools
import threading

__all__ = ['BaseRegistry', 'Registry', 'RegistryGroup', 'DefaultRegistry', 'CallbackRegistry', 'LockRegistry',
           'ModuleRegistry', 'ModuleRegistryGroup']


class BaseRegistry(object):
    __FALLBACK_KEY__ = '__fallback__'

    def __init__(self):
        self._init_registry()

    def _init_registry(self):
        self._registry = dict()

    @property
    def fallback(self):
        return self._registry.get(self.__FALLBACK_KEY__, None)

    def set_fallback(self, value):
        self._registry[self.__FALLBACK_KEY__] = value
        return self

    def register(self, key, value):
        self._registry[key] = value
        return self

    def unregister(self, key):
        return self._registry.pop(key)

    def has(self, key):
        return key in self._registry

    def lookup(self, key, fallback=True, default=None):
        if fallback:
            fallback_value = self._registry.get(self.__FALLBACK_KEY__, default)
        else:
            fallback_value = default
        return self._registry.get(key, fallback_value)

    def keys(self):
        return list(self._registry.keys())

    def values(self):
        return list(self._registry.values())

    def items(self):
        return list(self._registry.items())


Registry = BaseRegistry


class DefaultRegistry(BaseRegistry):
    __base_class__ = dict

    def _init_registry(self):
        base_class = type(self).__base_class__
        self._registry = collections.defaultdict(base_class)

    def lookup(self, key, fallback=False, default=None):
        assert fallback is False and default is None
        return self._registry.get(key)


class CallbackRegistry(Registry):
    def _init_registry(self):
        super(CallbackRegistry, self)._init_registry()
        self._super_callback = None

    @property
    def super_callback(self):
        return self._super_callback

    def set_super_callback(self, callback):
        self._super_callback = callback
        return self

    @property
    def fallback_callback(self):
        return self.fallback

    def set_fallback_callback(self, callback):
        return self.set_fallback(callback)

    def dispatch(self, key, *args, **kwargs):
        if self._super_callback is not None:
            return self._super_callback(key, *args, **kwargs)
        return self.dispatch_direct(key, *args, **kwargs)

    def dispatch_direct(self, key, *args, **kwargs):
        callback = self.lookup(key, fallback=False)
        if callback is None:
            if self.fallback_callback is None:
                raise ValueError(f'Unknown callback entry "{key}"')
            return self.fallback_callback(key, *args, **kwargs)
        return callback(*args, **kwargs)


class LockRegistry(DefaultRegistry):
    __base_class__ = threading.Lock

    def synchronized(self, key):
        return self.lookup(key)


class RegistryGroup(object):
    __base_class__ = Registry

    def __init__(self):
        self._init_registry_group()

    def _init_registry_group(self):
        base_class = type(self).__base_class__
        self._registry_group = collections.defaultdict(base_class)

    def register(self, registry_name, key, value):
        return self._registry_group[registry_name].register(key, value)

    def lookup(self, registry_name, key, fallback=True, default=None):
        return self._registry_group[registry_name].lookup(key, fallback, default)

    def registered_registry_keys(self, registry_name):
        return self._registry_group.get(registry_name).keys()

    def registered_registry_names(self):
        return self._registry_group.keys()


class ModuleRegistry(Registry):
    def register_module(self, name_or_module=None):
        if callable(name_or_module):
            super(ModuleRegistry, self).register(name_or_module.__name__, name_or_module)
        else:
            def wrapper(module):
                super(ModuleRegistry, self).register(name_or_module, module)
                return module

            return wrapper

    def build(self, name, *args, **kwargs):
        module = self.lookup(name)
        if module is None:
            raise ValueError(f"Module {name} is not registered")
        return module(*args, **kwargs)


class ModuleRegistryGroup(RegistryGroup):
    __base_class__ = ModuleRegistry

    @classmethod
    def get_module_type(cls, registry_name):
        module_type = registry_name.split('/')
        if len(module_type) > 1:
            module_type = '/'.join([':'.join(map(str.capitalize, module_type[:-1])), module_type[-1].capitalize()])
        else:
            module_type = module_type[-1].capitalize()
        return module_type

    def build(self, registry_name, name, *args, **kwargs):
        module = self.lookup(registry_name, name, fallback=False)
        module_type = self.get_module_type(registry_name)
        if module is None:
            raise ModuleNotFoundError(f"{module_type} {name} is not registered")
        instance = module(*args, **kwargs)
        instance.__registry_name__ = name
        instance.__jsonstr__ = f"<{module_type}: {name}>"
        return instance

    def build_fn(self, registry_name, name, *args, **kwargs):
        fn = functools.partial(self.build, registry_name, name, *args, **kwargs)
        fn.__jsonstr__ = f"(Build {self.get_module_type(registry_name)}: {name}, args={args}, kwargs={kwargs})"
        return fn

    def register_module(self, registry_name, name_or_module=None, module=None):
        if module is not None:
            self._registry_group[registry_name].register(name_or_module, module)
            return module
        elif callable(name_or_module):
            self._registry_group[registry_name].register(name_or_module.__name__, name_or_module)
            return name_or_module
        elif name_or_module is not None:
            return functools.partial(self.register_module, registry_name, name_or_module)
        else:
            return functools.partial(self.register_module, registry_name)
