import contextlib
import time

import torch.distributed

from pytools.pyutils.misc.string.helpers import remove_prefix

__all__ = ["DataMeter", "TimeMeter"]


class DataMeter(object):
    def __init__(self, default_device):
        self._statistics = dict()
        self._device = default_device

    def record(self, name, value, count, reduction):
        if isinstance(value, dict):
            for k, v in value.items():
                self.record(name + "/" + k, v, count, reduction[k])
        else:
            if isinstance(value, torch.Tensor):
                value = value.detach().clone().to(self._device)
            else:
                value = torch.scalar_tensor(value, dtype=torch.float, device=self._device)
            count = torch.scalar_tensor(count, dtype=torch.int64, device=self._device)

            if reduction == "mean":
                value.mul_(count)

            if name not in self._statistics:
                self._statistics[name] = (count, value)
            else:
                self._statistics[name][0].add_(count)
                self._statistics[name][1].add_(value)

    def synchronize(self, comm_group):
        futures = []
        for name in self._statistics:
            futures.append(torch.distributed.all_reduce(self._statistics[name][0], group=comm_group, async_op=True))
            futures.append(torch.distributed.all_reduce(self._statistics[name][1], group=comm_group, async_op=True))
        for future in futures:
            future.wait()

    def average(self, name):
        return self._statistics[name][1] / self._statistics[name][0]

    def summation(self, name):
        return self._statistics[name][1].clone()

    def reset(self, name=None):
        if name is None:
            self._statistics.clear()
        else:
            self._statistics.pop(name, None)

    def average_dict_with_prefix(self, prefix, use_sub_name=False):
        result = dict()
        for name in self._statistics:
            if name.startswith(prefix):
                if use_sub_name:
                    name = remove_prefix(name, prefix)
                result.setdefault(name, self.average(name))
        return result

    def fields(self):
        return list(self._statistics.keys())


class TimeMeter(object):
    def __init__(self):
        self._statistics = dict()

    @contextlib.contextmanager
    def record(self, name):
        self.before(name)
        try:
            yield
        finally:
            self.finish(name)

    def before(self, name):
        self._statistics.setdefault(name, float())
        self._statistics[name] -= time.time()

    def finish(self, name):
        self._statistics[name] += time.time()

    def reset(self, name=None):
        if name is None:
            self._statistics.clear()
        else:
            self._statistics.pop(name, None)
