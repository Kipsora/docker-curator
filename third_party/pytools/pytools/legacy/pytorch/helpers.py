__all__ = ['set_torch_seed', 'get_global_state', 'set_global_state']

import random

import numpy
import torch


def set_torch_seed(seed, use_stable_torch=True):
    import torch.backends.cudnn
    torch.manual_seed(seed)
    if use_stable_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_global_state():
    return {
        "global_state": {
            "torch": torch.random.get_rng_state(),
            "numpy": numpy.random.get_state(),
            "python": random.getstate()
        }
    }


def set_global_state(state):
    state = state['global_state']
    torch.random.set_rng_state(state['torch'])
    random.setstate(state['python'])
    numpy.random.set_state(state['numpy'])
