from typing import Dict, Optional

import torchvision

__all__ = ['ImageFolder']


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, class_indices: Optional[Dict[str, int]] = None):
        super().__init__(root, transform, target_transform)

        self._target_mapping = None
        if class_indices is not None:
            self._target_mapping = dict()
            for class_name in self.classes:
                if class_name not in class_indices:
                    raise ValueError(f"Class {class_name} is not found in class_indices")
                self._target_mapping[self.class_to_idx[class_name]] = class_indices[class_name]

    def __getitem__(self, item):
        source, target = super(ImageFolder, self).__getitem__(item)
        if self._target_mapping:
            target = self._target_mapping[target]
        return {'source': source, 'target': target}
