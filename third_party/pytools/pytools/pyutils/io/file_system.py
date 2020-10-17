import os
import shutil
from typing import IO

__all__ = ['get_extname', 'remove', 'as_file_descriptor', 'mkdir', 'get_relative_path', 'ensure_directory',
           'get_absolute_path', 'is_subdirectory', 'simplify_path', 'get_path_dirs']


def get_extname(path):
    return os.path.splitext(path)[1]


def remove(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def as_file_descriptor(instance: str or IO, **kwargs):
    if isinstance(instance, str):
        return open(instance, **kwargs)
    return instance


def mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


def get_relative_path(target, source='.', use_real_path=False):
    target = get_absolute_path(target, use_real_path)
    source = get_absolute_path(source, use_real_path)
    return os.path.relpath(target, source)


def get_absolute_path(path, use_real_path=False):
    return os.path.realpath(path) if use_real_path else os.path.abspath(path)


def is_subdirectory(path, parent_path=".", use_real_path=False):
    path = get_absolute_path(path, use_real_path)
    parent_path = get_absolute_path(parent_path, use_real_path)
    return path.startswith(parent_path)


def simplify_path(target, source='.', use_real_path=False):
    relative_path = get_relative_path(target, source, use_real_path)
    absolute_path = get_absolute_path(target, use_real_path)
    if len(relative_path) < len(absolute_path):
        return relative_path
    else:
        return absolute_path


def ensure_directory(path, create_if_not_exist=True):
    if not os.path.exists(path):
        if create_if_not_exist:
            os.makedirs(path)
        else:
            raise RuntimeError(f"Directory {path} does not exist")
    elif not os.path.isdir(path):
        raise RuntimeError(f"Path {path} is not a directory")
    return path


def get_path_dirs(path):
    result = []
    while path:
        parent_path, folder = os.path.split(path)
        result.append(folder)
        if path == parent_path:
            break
        path = parent_path

    result.reverse()
    return result
