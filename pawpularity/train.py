
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from pawpularity.augmentations import Augmentation
from pawpularity.config import Config, ResizerConfig
from pawpularity.datasets import PawModule, ResizerModule
from pawpularity.models import Model, ResizerModel


def train_main():
    config = Config()
    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)

    df_path = config.root_df
    img_path = config.root_img
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    folds = config.n_splits

    for fold in range(folds):
        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        val_df = df.loc[df["fold"] == fold].reset_index(drop=True)
        datamodule = PawModule(train_df, val_df, Augmentation, config)
        model = Model(config)
        earystopping = EarlyStopping(
            monitor="val_loss", verbose=config.verbose, patience=config.patience)
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


def resizer_train_main():
    config = ResizerConfig()
    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)

    df_path = config.root_df
    img_path = config.root_img
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    folds = config.n_splits

    for fold in range(folds):
        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        val_df = df.loc[df["fold"] == fold].reset_index(drop=True)
        datamodule = ResizerModule(train_df, val_df, Augmentation, config)
        model = ResizerModel(config)
        earystopping = EarlyStopping(
            monitor="val_acc", verbose=config.verbose, patience=config.patience)
        lr_monitor = callbacks.LearningRateMonitor()

        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_acc",
            monitor="val_acc",
            save_top_k=1,
            mode="max",
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
