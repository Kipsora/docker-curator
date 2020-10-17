import torch

from pytools.legacy.pytorch import curator
from pytools.legacy.pytorch.criterion.objectives import Objective

__all__ = []


def torch_loss_wrapper(torch_class):
    class WrappedClass(Objective):
        def __init__(self, **kwargs):
            super().__init__()
            self._objective = torch_class(**kwargs)

        @property
        def reduction(self):
            return {self.__OUTPUT_NAME__: self._objective.reduction}

        def forward(self, outputs, targets):
            return {self.__OUTPUT_NAME__: self._objective(outputs, targets)}

        def __repr__(self):
            return repr(self._objective)

    return WrappedClass


curator.register_module(
    'criterion/objective', 'AdaptiveLogSoftmaxWithLoss',
    torch_loss_wrapper(torch.nn.AdaptiveLogSoftmaxWithLoss)
)
curator.register_module('criterion/objective', 'BCELoss', torch_loss_wrapper(torch.nn.BCELoss))
curator.register_module('criterion/objective', 'BCEWithLogitsLoss', torch_loss_wrapper(torch.nn.BCEWithLogitsLoss))
curator.register_module('criterion/objective', 'CrossEntropyLoss', torch_loss_wrapper(torch.nn.CrossEntropyLoss))
curator.register_module('criterion/objective', 'CosineEmbeddingLoss', torch_loss_wrapper(torch.nn.CosineEmbeddingLoss))
curator.register_module('criterion/objective', 'CTCLoss', torch_loss_wrapper(torch.nn.CTCLoss))
curator.register_module('criterion/objective', 'HingeEmbeddingLoss', torch_loss_wrapper(torch.nn.HingeEmbeddingLoss))
curator.register_module('criterion/objective', 'KLDivLoss', torch_loss_wrapper(torch.nn.KLDivLoss))
curator.register_module('criterion/objective', 'L1Loss', torch_loss_wrapper(torch.nn.L1Loss))
curator.register_module('criterion/objective', 'MSELoss', torch_loss_wrapper(torch.nn.MSELoss))
curator.register_module('criterion/objective', 'MarginRankingLoss', torch_loss_wrapper(torch.nn.MarginRankingLoss))
curator.register_module('criterion/objective', 'MultiMarginLoss', torch_loss_wrapper(torch.nn.MultiMarginLoss))
curator.register_module('criterion/objective', 'MultiLabelMarginLoss', torch_loss_wrapper(torch.nn.MultiLabelMarginLoss))
curator.register_module(
    'criterion/objective', 'MultiLabelSoftMarginLoss',
    torch_loss_wrapper(torch.nn.MultiLabelSoftMarginLoss)
)
curator.register_module('criterion/objective', 'NLLLoss', torch_loss_wrapper(torch.nn.NLLLoss))
curator.register_module('criterion/objective', 'NLLLoss2d', torch_loss_wrapper(torch.nn.NLLLoss2d))
curator.register_module('criterion/objective', 'PoissonNLLLoss', torch_loss_wrapper(torch.nn.PoissonNLLLoss))
curator.register_module('criterion/objective', 'SoftMarginLoss', torch_loss_wrapper(torch.nn.SoftMarginLoss))
curator.register_module('criterion/objective', 'SmoothL1Loss', torch_loss_wrapper(torch.nn.SmoothL1Loss))
curator.register_module('criterion/objective', 'TripletMarginLoss', torch_loss_wrapper(torch.nn.TripletMarginLoss))
