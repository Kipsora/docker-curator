import torchvision

from pytools.legacy.pytorch import curator

__all__ = []


curator.register_module('dataset/transform', torchvision.transforms.ColorJitter)
curator.register_module('dataset/transform', torchvision.transforms.CenterCrop)
curator.register_module('dataset/transform', torchvision.transforms.FiveCrop)
curator.register_module('dataset/transform', torchvision.transforms.Grayscale)
curator.register_module('dataset/transform', torchvision.transforms.Lambda)
curator.register_module('dataset/transform', torchvision.transforms.Normalize)
curator.register_module('dataset/transform', torchvision.transforms.Pad)
curator.register_module('dataset/transform', torchvision.transforms.RandomAffine)
curator.register_module('dataset/transform', torchvision.transforms.RandomApply)
curator.register_module('dataset/transform', torchvision.transforms.RandomChoice)
curator.register_module('dataset/transform', torchvision.transforms.RandomCrop)
curator.register_module('dataset/transform', torchvision.transforms.RandomErasing)
curator.register_module('dataset/transform', torchvision.transforms.RandomGrayscale)
curator.register_module('dataset/transform', torchvision.transforms.RandomHorizontalFlip)
curator.register_module('dataset/transform', torchvision.transforms.RandomPerspective)
curator.register_module('dataset/transform', torchvision.transforms.RandomOrder)
curator.register_module('dataset/transform', torchvision.transforms.RandomRotation)
curator.register_module('dataset/transform', torchvision.transforms.RandomResizedCrop)
curator.register_module('dataset/transform', torchvision.transforms.RandomSizedCrop)
curator.register_module('dataset/transform', torchvision.transforms.RandomVerticalFlip)
curator.register_module('dataset/transform', torchvision.transforms.Resize)
curator.register_module('dataset/transform', torchvision.transforms.Scale)
curator.register_module('dataset/transform', torchvision.transforms.ToTensor)
