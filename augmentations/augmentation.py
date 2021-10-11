import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


class TrainAugmentation:

    def __init__(self,
                 config):
        self.augmentation = A.Compose([A.Resize(height=config.input_dim[0], width=config.input_dim[1], p=1),
                                       A.HorizontalFlip(),
                                       A.VerticalFlip(),
                                       A.RandomBrightnessContrast(),
                                       A.RandomCrop(),
                                       A.Normalize(config.image_mean, config.image_std),
                                       ToTensorV2()], p=1)

    def __call__(self, x: np.ndarray): 
        transform = self.augmentation(image=x)
        return transform['image']

class ValAugmentation: 

    def __init__(self, 
                 config):
        self.augmentation = A.Compose([A.Resize(height=config.input_dim[0], width=config.input_dim[1]),
                                       A.Normalize(config.image_mean, config.image_std),
                                       ToTensorV2()], p=1)
    
    def __call__(self, x: np.ndarray):
        transform = self.augmentation(image=x)
        return transform['image']

class Augmentation:

    def __init__(self, config):
        self.config = config
    
    def get_augmentation_by_mode(self, mode):
        if mode == 'train':
            return TrainAugmentation(self.config)
        return ValAugmentation(self.config)
