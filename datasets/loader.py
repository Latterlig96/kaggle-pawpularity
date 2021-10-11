from pytorch_lightning import LightningDataModule
import cv2
from torch.utils.data import Dataset, DataLoader


class PawDataset(Dataset):

    def __init__(self,
               df,
               transform,
               cfg):
        self.X = df["Id"].values
        self.transform = transform
        self.y = None if "Pawpularity" not in df.keys() else df["Pawpularity"].values

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image_path = self.X[idx]
        image = cv2.imread(image_path)
        image = self.transform(image)
        if self.y is not None:
            label = self.y[idx]
            return image, label
        return image

class PawModule(LightningDataModule):

    def __init__(self,
                train_df,
                val_df,
                transform,
                cfg):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.transform = transform
        self.cfg = cfg
    
    def _create_loader(self, train: bool = True):
        if train:
            return PawDataset(self.train_df, self.transform, self.cfg)
        return PawDataset(self.val_df, self.transform, self.cfg)
    
    def train_dataloader(self):
        dataset = self._create_loader(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self._create_loader(False)
        return DataLoader(dataset, **self._cfg.val_loader)