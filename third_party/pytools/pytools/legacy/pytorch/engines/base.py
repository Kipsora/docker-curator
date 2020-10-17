import abc
import collections
import typing

__all__ = ['Engine']

HookKeySet = collections.namedtuple("HookKeySet", ['before_hooks', 'finish_hooks'])


class Engine(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self._hooks: typing.Dict[str, 'HookKeySet'] = dict()

    def hook_keys(self):
        return tuple(self._hooks.keys())

    def has_hook_key(self, key):
        return key in self._hooks

    def _register_hook_key(self, key):
        self._hooks.setdefault(key, HookKeySet(list(), list()))

    def _unregister_hook_key(self, key):
        self._hooks.pop(key, None)

    def register_before_hook(self, key, hook):
        self._hooks[key].before_hooks.append(hook)

    def unregister_before_hook(self, key, hook):
        self._hooks[key].before_hooks.remove(hook)

    def register_finish_hook(self, key, hook):
        self._hooks[key].finish_hooks.append(hook)

    def unregister_finish_hook(self, key, hook):
        self._hooks[key].finish_hooks.remove(hook)

    def _before(self, key):
        for hook in self._hooks[key].before_hooks:
            hook(self)

    def _finish(self, key):
        for hook in self._hooks[key].finish_hooks:
            hook(self)

    def mount(self, object_or_dict):
        if object_or_dict is None:
            return
        if isinstance(object_or_dict, dict):
            for k, v in object_or_dict.items():
                if not isinstance(k, str) or not callable(v):
                    continue

                if k.startswith('before_') and self.has_hook_key(k[7:]):
                    self.register_before_hook(k[7:], v)
                if k.startswith('finish_') and self.has_hook_key(k[7:]):
                    self.register_finish_hook(k[7:], v)
        else:
            self.mount({k: getattr(object_or_dict, k) for k in dir(object_or_dict)})
