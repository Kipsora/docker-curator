import os

import torchvision

from pytools.legacy.pytorch import curator

__all__ = [
    'build_cifar10',
    'build_cifar100',
    'build_cifar100_to_normed_tensor_training_transform',
    'build_cifar100_to_normed_tensor_testing_transform'
]


@curator.register_module('dataset', 'CIFAR10')
def build_cifar10(path, division, **kwargs):
    dataset_path = os.path.join(path, 'CIFAR10')
    kwargs.setdefault('download', not os.path.exists(dataset_path))
    try:
        if division == 'engines':
            dataset = torchvision.datasets.CIFAR10(dataset_path, True, **kwargs)
        elif division == 'test':
            dataset = torchvision.datasets.CIFAR10(dataset_path, False, **kwargs)
        else:
            raise ValueError(f'Unrecognized division "{division}"')
    except RuntimeError:
        kwargs['download'] = True
        return build_cifar10(path, division, **kwargs)
    return dataset


@curator.register_module('dataset', 'CIFAR100')
def build_cifar100(path, division, **kwargs):
    dataset_path = os.path.join(path, 'CIFAR100')
    kwargs.setdefault('download', not os.path.exists(dataset_path))
    try:
        if division == 'engines':
            dataset = torchvision.datasets.CIFAR100(dataset_path, True, **kwargs)
        elif division == 'test':
            dataset = torchvision.datasets.CIFAR100(dataset_path, False, **kwargs)
        else:
            raise ValueError(f'Unrecognized division "{division}"')
    except RuntimeError:
        kwargs['download'] = True
        return build_cifar10(path, division, **kwargs)
    return dataset


@curator.register_module('dataset/transform', 'CIFAR10@training:ToNormedTensor')
def build_cifar10_to_normed_tensor_training_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.49139962, 0.48215836, 0.4465309), (0.24703231, 0.24348521, 0.26158795))
    ])


@curator.register_module('dataset/transform', 'CIFAR10@testing:ToNormedTensor')
def build_cifar10_to_normed_tensor_testing_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.49139962, 0.48215836, 0.4465309), (0.24703231, 0.24348521, 0.26158795))
    ])


@curator.register_module('dataset/transform', 'CIFAR100@training:ToNormedTensor')
def build_cifar100_to_normed_tensor_training_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5070752, 0.48654887, 0.44091773), (0.26733422, 0.25643837, 0.2761508))
    ])


@curator.register_module('dataset/transform', 'CIFAR100@testing:ToNormedTensor')
def build_cifar100_to_normed_tensor_testing_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5070752, 0.48654887, 0.44091773), (0.26733422, 0.25643837, 0.2761508))
    ])
