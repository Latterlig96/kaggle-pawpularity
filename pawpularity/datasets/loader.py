import cv2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize
import torch.nn.functional as F


class ResizerDataset(Dataset):

    def __init__(self, df, mode, transform, cfg):
        self.X = df['Id'].values
        self.transform = transform
        self.mode = mode
        self.cfg = cfg

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_path = self.X[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        if self.mode == 'train' or self.mode == 'val':
            image = resize(image, size=(
                self.cfg.input_image_size[0], self.cfg.input_image_size[1]))
            target = resize(image, size=(
                self.cfg.target_size[0], self.cfg.target_size[1]))
            return image, target
        return image


class ResizerModule(LightningDataModule):

    def __init__(self,
                 train_df,
                 val_df,
                 transform,
                 cfg):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.transform = transform(cfg)
        self.cfg = cfg

    def _create_loader(self, train: bool = True):
        if train:
            return ResizerDataset(self.train_df, 'train', self.transform.get_augmentation_by_mode('resizer'), self.cfg)
        return ResizerDataset(self.val_df, 'val', self.transform.get_augmentation_by_mode('resizer-val'), self.cfg)

    def train_dataloader(self):
        dataset = self._create_loader(True)
        return DataLoader(dataset, **self.cfg.train_loader)

    def val_dataloader(self):
        dataset = self._create_loader(False)
        return DataLoader(dataset, **self.cfg.val_loader)


class PawDataset(Dataset):

    def __init__(self,
                 df,
                 transform,
                 cfg):
        self.X = df["Id"].values
        self.transform = transform
        self.y = None if "Pawpularity" not in df.keys(
        ) else df["Pawpularity"].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_path = self.X[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        if self.y is not None:
            label = self.y[idx]
            return image, label
        return image
    
    @classmethod
    def as_dataloader(cls, df, transform, cfg, **kwargs):
        _cls = cls(df, transform , cfg)
        return DataLoader(_cls, **kwargs)
    

class PawModule(LightningDataModule):

    def __init__(self,
                 train_df,
                 val_df,
                 transform,
                 cfg):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.transform = transform(cfg)
        self.cfg = cfg

    def _create_loader(self, train: bool = True):
        if train:
            return PawDataset(self.train_df, self.transform.get_augmentation_by_mode('train'), self.cfg)
        return PawDataset(self.val_df, self.transform.get_augmentation_by_mode('val'), self.cfg)

    def train_dataloader(self):
        dataset = self._create_loader(True)
        return DataLoader(dataset, **self.cfg.train_loader)

    def val_dataloader(self):
        dataset = self._create_loader(False)
        return DataLoader(dataset, **self.cfg.val_loader)
