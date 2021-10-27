import albumentations as A
import numpy as np
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.augmentations.transforms import ColorJitter, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2


class TrainAugmentation:

    def __init__(self,
                 config):
        self.augmentation = A.Compose([A.HorizontalFlip(),
                                       A.RandomRotate90(),
                                       ColorJitter(**config.augmentation['color_jitter']),
                                       Affine(**config.augmentation['affine']),
                                       A.Resize(height=config.image_size[0], width=config.image_size[1]),
                                       A.Normalize(config.image_mean, config.image_std),
                                       ToTensorV2()])

    def __call__(self, x: np.ndarray): 
        transform = self.augmentation(image=x)
        return transform['image']

class ValAugmentation: 

    def __init__(self, 
                 config):
        self.augmentation = A.Compose([A.Resize(height=config.image_size[0], width=config.image_size[1]),
                                       A.Normalize(config.image_mean, config.image_std),
                                       ToTensorV2()])
    
    def __call__(self, x: np.ndarray):
        transform = self.augmentation(image=x)
        return transform['image']

class ResizerAugmentation:

    def __init__(self, config):
        self.augmentation = A.Compose([HorizontalFlip(),
                                       A.Normalize(config.image_mean, config.image_std),
                                       ToTensorV2()])
    
    def __call__(self, x: np.ndarray):
        transform = self.augmentation(image=x)
        return transform['image']

class ResizerValAugmentation:

    def __init__(self, config):
        self.augmentation = A.Compose([A.Resize(height=config.input_image_size[0], width=config.input_image_size[1]),
                                       A.Normalize(config.image_mean, config.image_std),
                                       ToTensorV2()])
    
    def __call__(self, x: np.ndarray):
        transform = self.augmentation(image=x)
        return transform['image']

class Augmentation:

    def __init__(self, config):
        self.config = config
    
    def get_augmentation_by_mode(self, mode):
        if mode == 'train':
            return TrainAugmentation(self.config)
        elif mode == 'resizer':
            return ResizerAugmentation(self.config)
        elif mode == 'resizer-val':
            return ResizerValAugmentation(self.config)
        return ValAugmentation(self.config)
