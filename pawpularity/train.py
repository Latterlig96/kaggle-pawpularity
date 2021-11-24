
import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from pawpularity.augmentations import Augmentation
from pawpularity.config import Config, EnsembleConfig, ResizerConfig
from pawpularity.datasets import PawDataset, PawModule, ResizerModule
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


def ensemble_train_main():
    import cuml
    import scipy.optimize as optimize

    ensemble_config = EnsembleConfig()
    df_path = ensemble_config.root_df
    img_path = ensemble_config.root_img
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    folds = ensemble_config.n_splits

    for fold in range(folds):
        print(f"Training fold {fold}")

        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        val_df = df.loc[df["fold"] == fold].reset_index(drop=True)
        
        embeds = np.zeros((len(train_df), len(ensemble_config.first_level_models)))

        dataloader = PawDataset.as_dataloader(train_df,
                                                Augmentation.get_augmentation_by_mode('train'),
                                                ensemble_config,
                                                **ensemble_config.data_loader)
                                                  
        for idx, name in enumerate(ensemble_config.first_level_models):
            config = Config.load_config_class(os.path.join(name, 'hyparams.yaml'))
            model = Model(config).load_from_checkpoint(os.path.join(name, fold, 'checkpoints', 'best_loss.ckpt'))

            trainer = pl.Trainer()

            train_predictions = trainer.predict(model, dataloader, return_predictions=True)

            embeds[:, idx] = train_predictions

        clf = cuml.SVR(C=20.0)

        clf.fit(embeds.astype('float32'), train_df['Pawpularity'].values.astype('int32'))

        pickle.dump(clf, open(f'SVR_FOLD_{idx}.pkl'))

        dataloader = PawDataset.as_dataloader(val_df,
                                              Augmentation.get_augmentation_by_mode('tta'),
                                              ensemble_config,
                                              **ensemble_config.data_loader)
        
        val_embeds = np.zeros((len(val_df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):

            tta_preds = np.zeros((len(val_df), ensemble_config.tta_steps))
            config = Config.load_config_class(os.path.join(name, 'hyparams.yaml'))
            model = Model(config).load_from_checkpoint(os.path.join(name, fold, 'checkpoints', 'best_loss.ckpt'))

            for idx in range(ensemble_config.tta_steps):
                trainer = pl.Trainer()

                train_predictions = trainer.predict(model, dataloader, return_predictions=True)

                tta_preds[:, idx] = train_predictions

            val_embeds[:, idx] = np.apply_along_axis(np.mean, axis=1, arr=tta_preds)

        clf_preds = clf.predict(val_embeds)

        targets = val_df['Pawpularity'].values

        nn_rmse = np.sqrt(np.mean(targets - val_embeds) ** 2)
        clf_rmse = np.sqrt(np.mean(targets - clf_preds) ** 2)

        print(f"NN RMSE: {nn_rmse}")
        print(f"Clf RMSE: {clf_rmse}")

        def minimize_rmse(x, *args):
            nn_preds = args[0][:, 0]
            clf_preds = args[0][:, 1]
            targets = args[0][:, 2]
            oof = (1-x)*nn_preds + x*clf_preds
            oof_rmse = np.sqrt(np.mean(targets - oof) ** 2)
            return oof_rmse
        
        result = optimize.minimize(minimize_rmse,
                                   x0=[0.5,], 
                                   args=(val_embeds, clf_preds, targets),
                                   bounds=((0, 1),),
                                   method='BFGS')
        
        best_oof_preds = (1-result.x)*val_embeds + result.x*clf_preds
        oof_rmse = np.sqrt(np.mean(targets-best_oof_preds)**2)

        print(f"Ensemble RMSE: {oof_rmse}")
