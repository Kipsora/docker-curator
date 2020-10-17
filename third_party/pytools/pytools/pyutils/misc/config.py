import argparse
import importlib
import os

from pytools.pyutils.io.file_system import get_path_dirs, get_relative_path, is_subdirectory, get_absolute_path
from pytools.pyutils.io.pretty import load_yaml
from pytools.pyutils.misc.nested import AttrListDictifier, DictOnlyMerger

__all__ = ["ConfigBuilder"]


class ConfigBuilder(object):
    def __init__(self, python_member="kwargs"):
        self._python_member = python_member
        self._config = dict()

        self._merger = DictOnlyMerger()
        self._dictifier = AttrListDictifier()

    @property
    def python_member(self):
        return self._python_member

    def load_python_config(self, path: str, use_parsed_path=True):
        try:
            current_path = list()
            parsed_path = self.get_parsed_path(path)
            for item in parsed_path:
                current_path.append(item)
                module = importlib.import_module(".".join(current_path))
                current_config = getattr(module, self._python_member, {})
                self._config = self._merger.merge(self._config, current_config)
            if use_parsed_path:
                self._config.setdefault("__parsed_path__", parsed_path)
            return self
        except ModuleNotFoundError as exception:
            raise RuntimeError(f"Cannot load python configuration from \"{path}\" "
                               f"because such module does not exist") from exception

    def load_from_args(self, args: argparse.Namespace):
        self._config = self._merger.merge(self._config, vars(args))
        return self

    @classmethod
    def parse_configs_path(cls, session_type: str):
        def compose_path(dirs, path):
            if not dirs:
                return path

            buffered_items = []
            for index, item in enumerate(dirs):
                buffered_items.append(item)

                is_path_possible = False
                if os.path.isdir(path):
                    for name in os.listdir(path):
                        if os.path.splitext(name)[0] == ".".join(buffered_items) or name == ".".join(buffered_items):
                            is_path_possible = True
                            break

                if is_path_possible:
                    result = compose_path(dirs[index + 1:], os.path.join(path, ".".join(buffered_items)))
                    if result:
                        return result
            return None

        configs_path = compose_path(session_type.split('.'), ".")
        if configs_path is None:
            raise ValueError(f"Cannot find any valid configs path from session type \"{session_type}\"")
        return configs_path

    def load_yaml_config(self, path: str, use_parsed_path=True):
        current_path = list()
        parsed_path = self.get_parsed_path(path)
        for item in parsed_path:
            current_path.append(item)

            if os.path.isfile(os.path.join(*current_path, "defaults.yml")):
                current_config = load_yaml(os.path.join(*current_path, "defaults.yml"))
            else:
                continue
            self._config = self._merger.merge(self._config, current_config)

        current_path = '/'.join(current_path)
        current_config = None
        if os.path.isfile(current_path):
            current_config = load_yaml(current_path)
        elif os.path.isfile(current_path + ".yml"):
            current_config = load_yaml(current_path + ".yml")
        if current_config is not None:
            self._config = self._merger.merge(self._config, current_config)

        if use_parsed_path:
            self._config.setdefault("__parsed_path__", parsed_path)
        return self

    def get_config(self):
        return self._dictifier.dictify(self._config)

    @classmethod
    def get_parsed_path(cls, path):
        if not is_subdirectory(path):
            raise ValueError("Only path in current directory can be parsed")

        path = get_absolute_path(path)
        if os.path.isfile(path):
            dirname, filename = os.path.split(path)
            path = os.path.join(dirname, os.path.splitext(filename)[0])
        path = get_relative_path(path)
        dirs = get_path_dirs(path)
        return dirs

    def set_config(self, key, value):
        self._config[key] = value
