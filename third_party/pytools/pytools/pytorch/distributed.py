import contextlib
import os

import numpy as np
import torch.distributed
import torch.utils.data

from pytools.pyutils.io.serialize import dumps_pickle, loads_pickle
from pytools.pyutils.misc.decorator import check_fn, check_failed_message

__all__ = [
    "activate",
    "register_extra_group",
    "unregister_extra_group",
    "is_distribute_activated",
    "is_distribute_enabled",
    "is_master",
    "is_local_master",
    "get_batch_size",
    "get_rank",
    "get_size",
    'broadcast'
]

_REGISTERED_EXTRA_GROUPS = set()
_IS_CONTEXT_ACTIVATED = False
_IS_DISTRIBUTED_ENABLED = False
_LOCAL_RANK = None
_STAT_GROUP = None


@check_failed_message("Distributed environment is not activated")
def is_distribute_activated():
    return _IS_CONTEXT_ACTIVATED


@check_failed_message("Distributed environment is not enabled")
def is_distribute_enabled():
    return is_distribute_activated() and _IS_DISTRIBUTED_ENABLED


@check_fn(is_distribute_enabled)
def register_extra_group(*args, **kwargs):
    global _REGISTERED_EXTRA_GROUPS
    group = torch.distributed.new_group(*args, **kwargs)
    _REGISTERED_EXTRA_GROUPS.add(group)
    return group


@check_fn(is_distribute_enabled)
def unregister_extra_group(group=None):
    global _REGISTERED_EXTRA_GROUPS
    if group is None:
        group = _REGISTERED_EXTRA_GROUPS.pop()
    else:
        _REGISTERED_EXTRA_GROUPS.remove(group)
    torch.distributed.destroy_process_group(group)


@check_fn(is_distribute_enabled)
def get_rank():
    return torch.distributed.get_rank()


@check_fn(is_distribute_enabled)
def get_size():
    return torch.distributed.get_world_size()


@check_fn(is_distribute_enabled)
def get_local_rank():
    return _LOCAL_RANK


def is_local_master():
    return (not is_distribute_enabled()) or (get_local_rank() == 0)


def is_master():
    return (not is_distribute_enabled()) or (get_rank() == 0)


def get_sampler(dataset, shuffle=True):
    if is_distribute_enabled():
        return torch.utils.data.distributed.DistributedSampler(dataset, get_size(), get_rank(), shuffle=shuffle)
    else:
        return None


def get_batch_size(batch_size):
    if not is_distribute_enabled():
        return batch_size
    size = get_size()
    return (batch_size // size) + int(get_rank() < batch_size % size)


@contextlib.contextmanager
def activate(is_distributed_enabled=False, local_rank=None, *args, **kwargs):
    global _IS_CONTEXT_ACTIVATED
    global _IS_DISTRIBUTED_ENABLED
    global _LOCAL_RANK
    global _STAT_GROUP

    if _IS_CONTEXT_ACTIVATED:
        raise RuntimeError("Distributed module is already activated")

    try:
        _IS_DISTRIBUTED_ENABLED = is_distributed_enabled
        _IS_CONTEXT_ACTIVATED = True
        if is_distributed_enabled:
            group = torch.distributed.init_process_group(*args, **kwargs)
            _LOCAL_RANK = int(local_rank) if local_rank else int(os.environ.get("LOCAL_RANK", get_rank()))

            yield group
        else:
            yield
    finally:
        while _REGISTERED_EXTRA_GROUPS:
            unregister_extra_group()

        if is_distributed_enabled:
            torch.distributed.destroy_process_group()

        _LOCAL_RANK = None
        _IS_DISTRIBUTED_ENABLED = False
        _IS_CONTEXT_ACTIVATED = False
        _STAT_GROUP = None


def broadcast(data, source=0, group=None):
    if not is_distribute_enabled() or get_size() == 1:
        if is_distribute_enabled() and (get_rank() != source or source != 0):
            raise RuntimeError(f"Unknown source rank {source}")
        return data

    if not is_distribute_enabled():
        raise RuntimeError(is_distribute_enabled.__failed_message__)

    if group is None:
        global _STAT_GROUP
        group = _STAT_GROUP = _STAT_GROUP or register_extra_group(backend='gloo')

    tensor = torch.ByteStorage.from_buffer(dumps_pickle(data))
    tensor = torch.ByteTensor(tensor)
    torch.distributed.broadcast(tensor, source, group=group)

    return loads_pickle(tensor.numpy().tobytes())


def all_reduce(data, group=None, use_async_op=False):
    if not is_distribute_enabled() or get_size() == 1:
        return list() if use_async_op else data

    if group is None:
        global _STAT_GROUP
        group = _STAT_GROUP = _STAT_GROUP or register_extra_group(backend='gloo')

    if isinstance(data, (int, np.integer)):
        data = torch.tensor(data, dtype=torch.int)
    elif isinstance(data, (float, np.floating)):
        data = torch.tensor(data, dtype=torch.float)
    elif isinstance(data, np.ndarray):
        data = torch.tensor(data)
    elif not isinstance(data, torch.Tensor):
        raise ValueError("Cannot convert data to torch Tensor")

    task = torch.distributed.all_reduce(data, group=group, async_op=True)
    if use_async_op:
        return [task]
    else:
        task.wait()
        return data


def barrier(group=None):
    if not is_distribute_enabled() or get_size() == 1:
        return

    if group is None:
        global _STAT_GROUP
        group = _STAT_GROUP = _STAT_GROUP or register_extra_group(backend='gloo')

    torch.distributed.barrier(group=group)
