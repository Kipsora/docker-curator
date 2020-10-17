from typing import Dict

import torch.utils.data

from pytools.pytorch.engines import Inferer, Engine
from pytools.pytorch.engines.callbacks import Callback

__all__ = ['InferOnIterators']


class InferOnIterators(Callback):
    def __init__(self, inferer: Inferer, eval_dict: Dict[str, torch.utils.data.DataLoader]):
        self._inferer = inferer
        self._eval_dict = eval_dict

    def after_epoch(self, engine: Engine, data_loader):
        for key, value in self._eval_dict.items():
            self._inferer.run(value, global_step=engine.global_step, dataset_name=key)
