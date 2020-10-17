import re

from pytools.pyutils.misc.decorator import deprecated

__all__ = ['AttrDict', 'AttrDictWrapper', 'nested_dictify', 'nested_merge']


class AttrDict(dict):
    def __getitem__(self, name):
        if name not in self and self.__contains__("__fallback__"):
            return self["__fallback__"]
        return super(AttrDict, self).__getitem__(name)

    def __setitem__(self, key, value):
        return super(AttrDict, self).__setitem__(key, value)

    def __contains__(self, item):
        return super(AttrDict, self).__contains__(item)

    def set_fallback(self, fallback):
        self["__fallback__"] = fallback

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError as exception:
            raise AttributeError(name) from exception

    __setattr__ = __setitem__
    __call__ = __getitem__


class AttrDictWrapper(object):
    def __init__(self, data: dict):
        object.__setattr__(self, '_data', data)

    def __getitem__(self, item):
        return dict.__getitem__(self._data, item)

    def __setitem__(self, key, value):
        return dict.__setitem__(self._data, key, value)

    def clear(self):
        self._data.clear()

    def __setstate__(self, state):
        assert isinstance(state, dict)
        object.__setattr__(self, '_data', state)

    def __getstate__(self):
        return self._data

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError as exception:
            raise AttributeError(f"{name} is an invalid attribute") from exception

    __setattr__ = __setitem__
    __call__ = __getitem__


@deprecated("This function is deprecated and will be removed in future releases. "
            "Use pytools.pyutils.misc.nested.nested_dictify instead.")
def nested_dictify(data, use_smart_nesting=True, dict_class=AttrDict):
    if isinstance(data, (list, tuple)):
        return data.__class__([nested_dictify(e, use_smart_nesting, dict_class) for e in data])
    elif isinstance(data, dict):
        items = []
        matched = True
        for k, v in data.items():
            if not re.match(r'[A-Za-z_][A-Za-z0-9_]*', k):
                matched = False
            items.append((k, nested_dictify(v, use_smart_nesting, dict_class)))
        if not use_smart_nesting or matched:
            return dict_class(items)
        else:
            return dict(items)
    else:
        return data


@deprecated("This function is deprecated and will be removed in future releases. "
            "Use pytools.pyutils.misc.nested.nested_merge instead.")
def nested_merge(a, b, dict_only=True):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return b
        result = a.__class__()
        for k, v in a.items():
            if k not in b:
                result.setdefault(k, v)
            else:
                result.setdefault(k, nested_merge(v, b[k], dict_only))
        for k, v in b.items():
            if k not in result:
                result.setdefault(k, v)
        return result
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or dict_only or len(a) != len(b):
            return b
        result = b.__class__()
        for i in range(len(b)):
            result.append(nested_merge(a[i], b[i], dict_only))
        return result
    else:
        return b
