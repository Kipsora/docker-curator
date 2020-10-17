import abc
import re

from pytools.pyutils.misc.attr_dict import AttrDict

__all__ = ['DictOnlyMerger', 'ListDictMerger', 'NestedMerger', 'NestedDictifier', 'AttrDictifier', 'AttrListDictifier',
           'nested_dictify', 'nested_merge']

from pytools.pyutils.misc.decorator import deprecated


class NestedMerger(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def merge(self, a, b):
        pass


class NestedDictifier(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def dictify(self, data):
        pass


class DictOnlyMerger(NestedMerger, metaclass=abc.ABCMeta):
    def _unhandled_handler(self, a, b, dict_class):
        return b

    def merge(self, a, b, dict_class=None):
        if isinstance(a, dict) and isinstance(b, dict):
            dict_class = dict_class or a.__class__
            result = dict_class()
            for k, v in a.items():
                if k not in b:
                    result.setdefault(k, v)
                else:
                    result.setdefault(k, self.merge(v, b[k], dict_class))
            for k, v in b.items():
                if k not in result:
                    result.setdefault(k, v)
            return result
        else:
            return self._unhandled_handler(a, b, dict_class)


class ListDictMerger(DictOnlyMerger):
    def _unhandled_handler(self, a, b, dict_class):
        if isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                return b
            result = b.__class__()
            for i in range(len(b)):
                result.append(self.merge(a[i], b[i], dict_class))
            return result
        else:
            return b


class AttrDictifier(NestedDictifier):
    def __init__(self, use_smart_attr=True):
        self._use_smart_attr = use_smart_attr

    def _unhandled_handler(self, data):
        return data

    def dictify(self, data):
        if isinstance(data, dict):
            items = []
            matched = True
            for k, v in data.items():
                if not re.match(r'[A-Za-z_][A-Za-z0-9_]*', k):
                    matched = False
                items.append((k, self.dictify(v)))
            if not self._use_smart_attr or matched:
                return AttrDict(items)
            else:
                return data.__class__(items)
        else:
            return self._unhandled_handler(data)


class AttrListDictifier(AttrDictifier):
    def _unhandled_handler(self, data):
        if isinstance(data, (list, tuple)):
            return data.__class__([self.dictify(e) for e in data])
        else:
            return data


@deprecated("This function is deprecated on 20200717 and will be removed in future releases. "
            "Use pytools.pyutils.misc.nested.NestedDictifier instead.")
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


@deprecated("This function is deprecated on 20200717 and will be removed in future releases. "
            "Use pytools.pyutils.misc.nested.NestedMerger instead.")
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
