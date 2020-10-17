import collections

__all__ = ['BiRelation', 'Bijection', 'Mapping']


class BiRelation(object):
    def __init__(self):
        self._init_relation()

    def _init_relation(self):
        self._sources_targets = collections.defaultdict(set)
        self._targets_sources = collections.defaultdict(set)

    def has_source(self, source):
        return source in self._sources_targets

    def has_target(self, target):
        return target in self._targets_sources

    def set(self, source, target):
        if target in self._sources_targets[source]:
            return False
        self._sources_targets[source].add(target)
        self._targets_sources[target].add(source)
        return True

    def unset(self, source=None, target=None):
        assert source is not None or target is not None
        if source is not None and source not in self._sources_targets:
            return False
        if target is not None and target not in self._targets_sources:
            return False
        sources = {source} if source is not None else self._targets_sources[target]
        targets = {target} if target is not None else self._sources_targets[source]
        for source in sources:
            for target in targets:
                self._sources_targets[source].remove(target)
                self._targets_sources[target].remove(source)
                if not self._sources_targets[source]:
                    self._sources_targets.pop(source)
                if not self._targets_sources[target]:
                    self._targets_sources.pop(target)
        return True

    def get_targets(self, source=None):
        if source is None:
            return set(self._targets_sources.keys())
        if source in self._sources_targets:
            return self._sources_targets[source].copy()
        else:
            return set()

    def get_sources(self, target=None):
        if target is None:
            return set(self._sources_targets.keys())
        if target in self._targets_sources:
            return self._targets_sources[target].copy()
        else:
            return set()


class Mapping(BiRelation):
    def _init_relation(self):
        self._sources_targets = dict()
        self._targets_sources = collections.defaultdict(set)

    def set(self, source, target, force=False):
        if not force and source in self._sources_targets:
            return False
        self._sources_targets[source] = target
        self._targets_sources[target].add(source)
        return True

    def unset(self, source=None, target=None):
        assert source is not None or target is not None
        if source is not None and source not in self._sources_targets:
            return False
        if target is not None and target not in self._targets_sources:
            return False
        sources = {source} if source is not None else {self._targets_sources[target]}
        targets = {target} if target is not None else self._sources_targets[source]
        for source in sources:
            for target in targets:
                self._sources_targets.pop(source)
                self._targets_sources[target].remove(source)
                if not self._targets_sources[target]:
                    self._targets_sources.pop(target)
        return True

    def unset_by_target(self, target, source=None):
        if source is not None:
            return self.unset(source, target)
        if target not in self._targets_sources:
            return False
        sources = self._targets_sources.pop(target)
        for source in sources:
            self._sources_targets[source].pop(target)
            if not self._sources_targets[source]:
                self._sources_targets.pop(source)
        return True

    def get_targets(self, source=None):
        if source is None:
            return self._targets_sources.keys()
        return self._sources_targets.get(source, default=None)


class Bijection(Mapping):
    def _init_relation(self):
        self._sources_targets = dict()
        self._targets_sources = dict()

    def set(self, source, target, force=False):
        if not force:
            if source in self._sources_targets:
                return False
            if target in self._targets_sources:
                return False
        self._sources_targets[source] = target
        self._targets_sources[target] = source
        return True

    def unset(self, source=None, target=None):
        assert source is not None or target is not None
        if source is not None and source not in self._sources_targets:
            return False
        if target is not None and target not in self._targets_sources:
            return False
        self._sources_targets.pop(source)
        self._targets_sources.pop(target)
        return True

    def get_sources(self, target=None):
        if target is None:
            return self._sources_targets.keys()
        return self._targets_sources.get(target, default=None)
