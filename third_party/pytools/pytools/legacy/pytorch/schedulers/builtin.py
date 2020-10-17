from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR

from pytools.legacy.pytorch import curator

__all__ = ['torch_scheduler_wrapper']


def torch_scheduler_wrapper(scheduler_class):
    class WrappedScheduler(scheduler_class):
        def finish_epoch(self, _):
            self.step()

    return WrappedScheduler


curator.register_module('scheduler', 'StepLR', torch_scheduler_wrapper(StepLR))
curator.register_module('scheduler', 'MultiStepLR', torch_scheduler_wrapper(MultiStepLR))
curator.register_module('scheduler', 'ReduceLROnPlateau', torch_scheduler_wrapper(ReduceLROnPlateau))
curator.register_module('scheduler', 'CyclicLR', torch_scheduler_wrapper(CyclicLR))
curator.register_module('scheduler', 'ExponentialLR', torch_scheduler_wrapper(ExponentialLR))
curator.register_module('scheduler', 'CosineAnnealingLR', torch_scheduler_wrapper(CosineAnnealingLR))
curator.register_module('scheduler', 'LambdaLR', torch_scheduler_wrapper(LambdaLR))
