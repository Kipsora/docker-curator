import logging
import random
from typing import Dict, Any

import numpy
import torch

__all__ = ['set_torch_seed', 'get_global_state', 'set_global_state', 'load_model_state']


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


def load_model_state(model: torch.nn.Module, state: Dict[str, Any], logger: logging.Logger):
    param_keys = set(model.state_dict().keys())
    proxy_state = dict()
    for key, value in state.items():
        is_proxy_key_found = False
        for param_key in param_keys:
            if key.endswith(param_key) or param_key.endswith(key):
                logger.debug(f'Model mapping: {key} => {param_key}')
                proxy_state.setdefault(param_key, value)
                if is_proxy_key_found:
                    raise RuntimeError("Find multiple available proxy parameter keys")
                is_proxy_key_found = True
        if not is_proxy_key_found:
            raise RuntimeError(f"Cannot find any available proxy parameter key for {key}")
    model.load_state_dict(proxy_state)
