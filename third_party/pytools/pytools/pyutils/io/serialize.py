from pytools.pyutils.io import io_registry_group
from pytools.pyutils.io.file_system import as_file_descriptor

__all__ = ['load_numpy', 'dump_numpy',
           'load_torch', 'dump_torch',
           'load_pickle', 'dump_pickle',
           'loads_pickle', 'dumps_pickle']


def _import_pickle():
    try:
        import cPickle as pickle
    except ModuleNotFoundError:
        import pickle
    return pickle


def load_pickle(file, **kwargs):
    pickle = _import_pickle()
    with as_file_descriptor(file, mode='rb') as reader:
        return pickle.load(reader, **kwargs)


def dump_pickle(data, file, **kwargs):
    pickle = _import_pickle()
    with as_file_descriptor(file, mode='wb') as writer:
        pickle.dump(data, writer, **kwargs)


def loads_pickle(data, **kwargs):
    pickle = _import_pickle()
    return pickle.loads(data, **kwargs)


def dumps_pickle(data, **kwargs):
    pickle = _import_pickle()
    return pickle.dumps(data, **kwargs)


try:
    import numpy as np


    def load_numpy(file, **kwargs):
        return np.load(file, **kwargs)


    def dump_numpy(file, data, **kwargs):
        return np.save(file, data, **kwargs)


    io_registry_group.register('load', '.npy', load_numpy)
    io_registry_group.register('dump', '.npy', dump_numpy)
except ModuleNotFoundError:
    pass

try:
    import torch


    def load_torch(file, **kwargs):
        import torch
        with as_file_descriptor(file, mode='rb') as reader:
            return torch.load(reader, **kwargs)


    def dump_torch(file, data, **kwargs):
        import torch
        with as_file_descriptor(file, mode='wb') as writer:
            return torch.save(data, writer, **kwargs)


    io_registry_group.register('load', '.pth', load_torch)
    io_registry_group.register('dump', '.pth', dump_torch)
except ModuleNotFoundError:
    pass

io_registry_group.register('load', '.pkl', load_pickle)
io_registry_group.register('dump', '.pkl', dump_pickle)
