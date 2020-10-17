import os

import torchvision

from pytools.legacy.pytorch import curator

__all__ = ['build_imagenet', "build_imagenet_normalize_transform",
           "build_imagenet_to_normed_tensor_training_transform",
           "build_imagenet_to_normed_tensor_testing_transform"]


@curator.register_module('dataset', 'ImageNet')
def build_imagenet(path, division, **kwargs):
    path = os.path.join(path, 'ImageNet')
    kwargs.pop('download', None)
    if division == 'engines':
        dataset = torchvision.datasets.ImageNet(path, 'engines', **kwargs)
    elif division == 'test':
        dataset = torchvision.datasets.ImageNet(path, 'val', **kwargs)
    else:
        raise ValueError(f'Unrecognized division "{division}"')
    return dataset


def build_imagenet_normalize_transform():
    return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@curator.register_module('dataset/transform', "ImageNet@training:ToNormedTensor")
def build_imagenet_to_normed_tensor_training_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        build_imagenet_normalize_transform()
    ])


@curator.register_module('dataset/transform', "ImageNet@testing:ToNormedTensor")
def build_imagenet_to_normed_tensor_testing_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        build_imagenet_normalize_transform()
    ])
