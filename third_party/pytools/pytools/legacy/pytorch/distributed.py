import contextlib
import os

import torch.distributed
import torch.utils.data

from pytools.pyutils.misc.decorator import check_fn, check_failed_message

__all__ = ["activate", "register_extra_group", "unregister_extra_group", "is_distribute_activated",
           "is_distribute_enabled", "is_master", "is_local_master", "get_batch_size",
           "get_rank", "get_size"]

_REGISTERED_EXTRA_GROUPS = set()
_IS_CONTEXT_ACTIVATED = False
_IS_DISTRIBUTED_ENABLED = False
_LOCAL_RANK = None


@check_failed_message("Distributed environment is not activated")
def is_distribute_activated():
    return _IS_CONTEXT_ACTIVATED


@check_fn(is_distribute_activated)
@check_failed_message("Distributed environment is not enabled")
def is_distribute_enabled():
    return _IS_DISTRIBUTED_ENABLED


def register_extra_group(*args, **kwargs):
    if not _IS_DISTRIBUTED_ENABLED:
        return None
    global _REGISTERED_EXTRA_GROUPS
    group = torch.distributed.new_group(*args, **kwargs)
    _REGISTERED_EXTRA_GROUPS.add(group)
    return group


def unregister_extra_group(group=None):
    if not is_distribute_enabled():
        return
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
