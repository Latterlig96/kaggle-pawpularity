
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
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    config = Config()
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
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True
    
    config = ResizerConfig()
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
    from cuml import SVR
    import scipy.optimize as optimize

    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    ensemble_config = EnsembleConfig()
    main_config = Config()
    df_path = ensemble_config.root_df
    img_path = ensemble_config.root_img
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    folds = ensemble_config.n_folds

    for fold in range(folds):
        print(f"Training fold {fold}")

        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        val_df = df.loc[df["fold"] == fold].reset_index(drop=True)
        
        train_embeds = np.zeros((len(ensemble_config.first_level_models), len(train_df), 384))

        train_dataloader = PawDataset.as_dataloader(train_df,
                                                Augmentation(ensemble_config).get_augmentation_by_mode('train'),
                                                ensemble_config,
                                                **ensemble_config.data_loader)
        
        for idx, name in enumerate(ensemble_config.first_level_models):
            model = Model.load_from_checkpoint(os.path.join(name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            trainer = pl.Trainer(**ensemble_config.trainer)

            train_predictions = trainer.predict(model, train_dataloader, return_predictions=True)

            train_embeds[idx, ...] = np.concatenate([pred['embeddings'] for pred in train_predictions], axis=0)

        classifiers = []

        for name, arr in zip(ensemble_config.first_level_models, train_embeds):
            clf = SVR(C=20.0, verbose=True)

            clf.fit(arr.astype('float32'), train_df['Pawpularity'].values.astype('int32'))

            with open(f'SVR_FOLD_{fold}_{name}.pkl', 'wb') as f:
                pickle.dump(clf, f)

            classifiers.append(clf)

        val_dataloader = PawDataset.as_dataloader(val_df,
                                              Augmentation(ensemble_config).get_augmentation_by_mode('tta'),
                                              ensemble_config,
                                              **ensemble_config.data_loader)
        
        val_embeds = np.zeros((len(ensemble_config.first_level_models), len(val_df), 384))
        val_targets = np.zeros((len(val_df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):

            tta_embed_preds = np.zeros((ensemble_config.tta_steps, len(val_df), 384))
            tta_preds = np.zeros((len(val_df), ensemble_config.tta_steps))
            model = Model.load_from_checkpoint(os.path.join(name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            for step in range(ensemble_config.tta_steps):
                trainer = pl.Trainer(**ensemble_config.trainer)

                valid_predictions = trainer.predict(model, val_dataloader, return_predictions=True)

                tta_embed_preds[step, ...] = np.concatenate([pred['embeddings'] for pred in valid_predictions], axis=0)
                tta_preds[:, step] = np.concatenate([pred['pred'] for pred in valid_predictions], axis=0)

            val_embeds[idx, ...] = np.apply_along_axis(np.mean, axis=0, arr=tta_embed_preds)
            val_targets[:, idx] = np.apply_along_axis(np.mean, axis=1, arr=tta_preds)
        
        for idx, (embeds, clf) in enumerate(zip(val_embeds, classifiers)):

            clf_preds = clf.predict(embeds.astype('float32'))

            targets = val_df['Pawpularity'].values

            nn_preds = val_targets[:, idx]

            nn_rmse = np.sqrt(np.mean((targets - nn_preds)**2.0))
            clf_rmse = np.sqrt(np.mean((targets - clf_preds)**2.0))

            print(f"NN RMSE: {nn_rmse}")
            print(f"Clf RMSE: {clf_rmse}")

            def minimize_rmse(x, *args):
                nn_preds = args[0]
                clf_preds = args[1]
                targets = args[2]
                oof = (1-x)*nn_preds + x*clf_preds
                oof_rmse = np.sqrt(np.mean(targets - oof) ** 2)
                return oof_rmse
            
            result = optimize.minimize(minimize_rmse,
                                    x0=[0.5,], 
                                    args=(nn_preds, clf_preds, targets),
                                    bounds=((0, 1),),
                                    method='L-BFGS-B')
            
            oof = (1-result.x)*np.array(nn_preds) + result.x*np.array(clf_preds)
            oof_rmse = np.sqrt(np.mean((targets - oof)**2.0))

            print(f"Ensemble RMSE: {oof_rmse}")
