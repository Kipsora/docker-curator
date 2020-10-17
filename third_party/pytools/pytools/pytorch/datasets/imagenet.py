import torchvision

__all__ = ['ImageNet', 'ImageNetBuilder']


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)

    def __getitem__(self, item):
        source, target = super(ImageNet, self).__getitem__(item)
        return {'source': source, 'target': target}


class ImageNetBuilder(object):
    def __init__(self, path):
        self._path = path

    def build_training_dataset(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return ImageNet(self._path, split='train', transform=transform)

    def build_validation_dataset(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return ImageNet(self._path, split='val', transform=transform)
