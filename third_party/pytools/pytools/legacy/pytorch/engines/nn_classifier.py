import torch
from torch.optim.optimizer import Optimizer

from pytools.legacy.pytorch import curator
from pytools.legacy.pytorch.criterion.base import Criterion
from pytools.legacy.pytorch.criterion.objectives import Objective
from pytools.legacy.pytorch.engines import Engine


__all__ = ['NNClassifierTrainer', 'NNClassifierInferer']


@curator.register_module('engine')
class NNClassifierTrainer(Engine):
    def __init__(self, model: torch.nn.Module, device, objective: Objective, optimizer: Optimizer):
        super().__init__()
        self._register_hook_key('epoch')
        self._register_hook_key('batch')
        self._register_hook_key('move_data')
        self._register_hook_key('forward')
        self._register_hook_key('forward_calc_logits')
        self._register_hook_key('forward_calc_loss')
        self._register_hook_key('backward')
        self._register_hook_key('backward_zero_grad')
        self._register_hook_key('backward_calc_grad')
        self._register_hook_key('backward_step_grad')

        self._model = model
        self._device = device
        self._objective = objective
        self._optimizer = optimizer

        self._epoch_idx = 0
        self._batch_idx = None
        self._batch_inputs = None
        self._batch_labels = None
        self._batch_logits = None
        self._batch_losses = None

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def objective(self):
        return self._objective

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def batch_idx(self):
        return self._batch_idx

    @property
    def batch_inputs(self):
        return self._batch_inputs

    @property
    def batch_labels(self):
        return self._batch_labels

    @property
    def batch_logits(self):
        return self._batch_logits

    @property
    def batch_losses(self):
        return self._batch_losses

    @property
    def batch_size(self):
        return len(self._batch_inputs)

    @property
    def epoch_idx(self):
        return self._epoch_idx

    def train_epoch(self, iterator):
        self._epoch_idx += 1

        self._model.train()

        self._before('epoch')
        for self._batch_idx, (self._batch_inputs, self._batch_labels) in enumerate(iterator):
            self._before('batch')

            self._before('move_data')
            self._batch_inputs = self._batch_inputs.to(self._device)
            self._batch_labels = self._batch_labels.to(self._device)
            self._finish('move_data')

            self._before('forward')
            self._before('forward_calc_logits')
            self._batch_logits = self._model(self._batch_inputs)
            self._finish('forward_calc_logits')
            self._before('forward_calc_loss')
            self._batch_losses = self._objective(self._batch_logits, self._batch_labels)
            self._finish('forward_calc_loss')
            self._finish('forward')

            self._before('backward')
            self._before('backward_zero_grad')
            self._optimizer.zero_grad()
            self._finish('backward_zero_grad')
            self._before('backward_calc_grad')
            self._batch_losses[self._objective.__OUTPUT_NAME__].backward()
            self._finish('backward_calc_grad')
            self._before('backward_step_grad')
            self._optimizer.step()
            self._finish('backward_step_grad')
            self._finish('backward')

            self._finish('batch')
        self._finish('epoch')

    def state_dict(self):
        return {
            'model': self._model.state_dict(),
            'objective': self._objective.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'epoch': self._epoch_idx
        }

    def restart(self, state):
        self._model.load_state_dict(state['model'])

    def restore(self, state):
        self.restart(state)
        self._objective.load_state_dict(state['objective'])
        self._optimizer.load_state_dict(state['optimizer'])
        self._epoch_idx = state['epoch']

    def train_until(self, epoch, iterator):
        while self._epoch_idx < epoch:
            self.train_epoch(iterator)


@curator.register_module('engine')
class NNClassifierInferer(Engine):
    def __init__(self, model, device, criterion: Criterion):
        super().__init__()
        self._register_hook_key('infer')
        self._register_hook_key('batch')
        self._register_hook_key('move_data')
        self._register_hook_key('forward')
        self._register_hook_key('forward_calc_logits')
        self._register_hook_key('forward_calc_loss')

        self._model = model
        self._device = device
        self._criterion = criterion

        self._batch_idx = None
        self._batch_inputs = None
        self._batch_labels = None
        self._batch_logits = None
        self._batch_losses = None
        self._dataset_name = None

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def batch_idx(self):
        return self._batch_idx

    @property
    def batch_inputs(self):
        return self._batch_inputs

    @property
    def batch_labels(self):
        return self._batch_labels

    @property
    def batch_logits(self):
        return self._batch_logits

    @property
    def batch_losses(self):
        return self._batch_losses

    @property
    def batch_size(self):
        return len(self._batch_inputs)

    @property
    def dataset_name(self):
        return self._dataset_name

    def restart(self, state):
        self._model.load_state_dict(state['model'])

    @torch.no_grad()
    def infer(self, iterator, name=None):
        self._model.eval()

        self._dataset_name = name

        self._before('infer')
        for self._batch_idx, (self._batch_inputs, self._batch_labels) in enumerate(iterator):
            self._before('batch')

            self._before('move_data')
            self._batch_inputs = self._batch_inputs.to(self._device)
            self._batch_labels = self._batch_labels.to(self._device)
            self._finish('move_data')

            self._before('forward')
            self._before('forward_calc_logits')
            self._batch_logits = self._model(self._batch_inputs)
            self._finish('forward_calc_logits')
            self._before('forward_calc_loss')
            self._batch_losses = self._criterion(self._batch_logits, self._batch_labels)
            self._finish('forward_calc_loss')
            self._finish('forward')

            self._finish('batch')
        self._finish('infer')

    def infer_multiple_datasets(self, iter_dict: dict):
        for name, iterator in iter_dict.items():
            self.infer(iterator, name)
