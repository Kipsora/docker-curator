import abc

from pytools.legacy.pytorch.criterion.base import Criterion


__all__ = ['Objective']


class Objective(Criterion, metaclass=abc.ABCMeta):
    __OUTPUT_NAME__ = 'loss'
