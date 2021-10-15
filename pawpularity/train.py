
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold

from pawpularity.augmentations import Augmentation
from pawpularity.config import Config
from pawpularity.datasets import PawModule
from pawpularity.models import Model


def train_main():
    config = Config()
    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)

    df_path = config.root + '/' + 'train.csv'
    img_path = config.root + '/' 'train'
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PawModule(train_df, val_df, Augmentation, config)
        model = Model(config)
        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()

        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )

        logger = TensorBoardLogger(config.model_name)
        
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epochs,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config.trainer,
        )
        
        trainer.fit(model, datamodule=datamodule)
