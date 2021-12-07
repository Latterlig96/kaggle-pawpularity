import os
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
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


def ensemble_train_stacking_wihout_second_level_fold():
    import scipy.optimize as optimize
    from cuml import SVR, Ridge

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
        logger.info(f"Training fold {fold}")

        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        val_df = df.loc[df["fold"] == fold].reset_index(drop=True)

        vit_train_embeds = np.zeros((len(train_df), 384))
        swin_train_embeds = np.zeros((len(train_df), 1024))

        train_dataloader = PawDataset.as_dataloader(train_df,
                                                    Augmentation(
                                                        ensemble_config).get_augmentation_by_mode('train'),
                                                    ensemble_config,
                                                    **ensemble_config.data_loader)

        for name in ensemble_config.first_level_models:
            main_config.model_name = name
            if os.path.exists(f'SVR_FOLD_{fold}_{name}.pkl'):
                logger.info(f"Skipping training {name}, not necessary")
                continue

            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            trainer = pl.Trainer(**ensemble_config.trainer)

            train_predictions = trainer.predict(
                model, train_dataloader, return_predictions=True)

            if name == 'ViTHybridSmallv2':
                vit_train_embeds = np.concatenate(
                    [pred['embeddings'] for pred in train_predictions], axis=0)
            else:
                swin_train_embeds = np.concatenate(
                    [pred['embeddings'] for pred in train_predictions], axis=0)

        classifiers = []

        for name in ensemble_config.first_level_models:
            if os.path.exists(f'SVR_FOLD_{fold}_{name}.pkl'):
                logger.info(
                    f"Skipping training SVR_FOLD_{fold}_{name}, model already exists")
                with open(f'SVR_FOLD_{fold}_{name}.pkl', 'rb') as clf:
                    clf = pickle.load(clf)
                    classifiers.append(clf)
                continue

            logger.info(f"Train SVR on {name} embeddings")

            clf = SVR(C=20.0, verbose=True)

            if name == 'ViTHybridSmallv2':
                clf.fit(vit_train_embeds.astype('float32'),
                        train_df['Pawpularity'].values.astype('int32'))
            else:
                clf.fit(swin_train_embeds.astype('float32'),
                        train_df['Pawpularity'].values.astype('int32'))

            with open(f'SVR_FOLD_{fold}_{name}.pkl', 'wb') as f:
                pickle.dump(clf, f)

            classifiers.append(clf)

        val_dataloader = PawDataset.as_dataloader(val_df,
                                                  Augmentation(
                                                      ensemble_config).get_augmentation_by_mode('tta'),
                                                  ensemble_config,
                                                  **ensemble_config.data_loader)

        vit_val_embeds = np.zeros((len(val_df), 384))
        swin_val_embeds = np.zeros((len(val_df), 1024))

        val_targets = np.zeros(
            (len(val_df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):
            main_config.model_name = name

            if name == 'ViTHybridSmallv2':
                tta_embed_preds = np.zeros(
                    (ensemble_config.tta_steps, len(val_df), 384))
            else:
                tta_embed_preds = np.zeros(
                    (ensemble_config.tta_steps, len(val_df), 1024))

            tta_preds = np.zeros((len(val_df), ensemble_config.tta_steps))
            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            for step in range(ensemble_config.tta_steps):
                trainer = pl.Trainer(**ensemble_config.trainer)

                valid_predictions = trainer.predict(
                    model, val_dataloader, return_predictions=True)

                tta_embed_preds[step, ...] = np.concatenate(
                    [pred['embeddings'] for pred in valid_predictions], axis=0)
                tta_preds[:, step] = np.concatenate(
                    [pred['pred'] for pred in valid_predictions], axis=0)

            if name == 'ViTHybridSmallv2':
                vit_val_embeds = np.apply_along_axis(
                    np.mean, axis=0, arr=tta_embed_preds)
            else:
                swin_val_embeds = np.apply_along_axis(
                    np.mean, axis=0, arr=tta_embed_preds)

            val_targets[:, idx] = np.apply_along_axis(
                np.mean, axis=1, arr=tta_preds)

        oof = np.zeros((len(val_df), len(ensemble_config.first_level_models)))

        for idx, (embeds, name, clf) in enumerate(zip([vit_val_embeds, swin_val_embeds], ensemble_config.first_level_models, classifiers)):

            clf_preds = clf.predict(embeds.astype('float32'))

            oof[:, idx] = clf_preds

            targets = val_df['Pawpularity'].values

            clf_rmse = np.sqrt(np.mean((targets - clf_preds)**2.0))

            logger.info(f"SVR fold: {fold} name: {name} RMSE: {clf_rmse}")

        targets = val_df['Pawpularity'].values

        train = oof[int(oof.shape[0] * ensemble_config.holdout_percent):, :]
        holdout_train = oof[:int(
            oof.shape[0] * ensemble_config.holdout_percent), :]
        train_targets = targets[int(
            targets.shape[0] * ensemble_config.holdout_percent):]
        holdout_targets = targets[:int(
            targets.shape[0] * ensemble_config.holdout_percent)]

        if os.path.exists(f'RIDGE_FOLD_{fold}.pkl'):
            logger.info(
                f'Skipping training RIDGE_FOLD_{fold}, model already exists')
            with open(f'RIDGE_FOLD_{fold}.pkl', 'rb') as clf:
                ridge_clf = pickle.load(clf)
            ridge_preds = ridge_clf.predict(holdout_train)
        else:
            ridge_clf = Ridge(alpha=1.0, solver='svd', verbose=True)
            ridge_clf.fit(train.astype('float32'),
                          train_targets.astype('int32'))
            with open(f'RIDGE_FOLD_{fold}.pkl', 'wb') as f:
                pickle.dump(ridge_clf, f)
            ridge_preds = ridge_clf.predict(holdout_train)

        vit_rmse = np.sqrt(np.mean((targets - val_targets[:, 0])**2.0))
        swin_rmse = np.sqrt(np.mean((targets - val_targets[:, 1])**2.0))

        ridge_rmse = np.sqrt(np.mean((holdout_targets - ridge_preds)**2.0))

        logger.info(f"VIT fold: {fold} RMSE: {vit_rmse}")
        logger.info(f"Swin fold: {fold} RMSE: {swin_rmse}")
        logger.info(f"Ridge fold: {fold} RMSE: {ridge_rmse}")

        def minimize_rmse(x, *args):
            vit_preds = args[0][:, 0]
            swin_preds = args[0][:, 1]
            targets = args[1]
            oof = (1-x)*vit_preds + x*swin_preds
            oof_rmse = np.sqrt(np.mean(targets - oof) ** 2)
            return oof_rmse

        def minimize_weighted_rmse(x, *args):
            vit_swin_rmse = args[0][:, 0]
            ridge_rmse = args[0][:, 1]
            oof = x[0]*vit_swin_rmse + x[1]*ridge_rmse
            final_oof = oof
            return final_oof

        result = optimize.differential_evolution(minimize_rmse,
                                                 args=(val_targets, targets),
                                                 bounds=((0, 1),))

        with open(f'VIT_SWIN_RESULT_{fold}.pkl', 'wb') as vit_swin_result:
            pickle.dump(result.x, vit_swin_result)

        vit_swin_ensemble = (
            1-result.x)*np.array(val_targets[:, 0]) + result.x*np.array(val_targets[:, 1])
        vit_swin_rmse = np.sqrt(np.mean((targets - vit_swin_ensemble)**2.0))
        logger.info(f"Vit/Swin Ensemble RMSE: {vit_swin_rmse}")

        def fconstr(x): return 1 - sum(x)
        constraints = ({'type': 'eq', 'fun': fconstr})
        result_weighted = optimize.differential_evolution(minimize_weighted_rmse,
                                                          args=(
                                                              vit_swin_rmse, ridge_rmse),
                                                          bounds=(
                                                              (0, 1), (0, 1)),
                                                          constraints=constraints)

        with open(f'WEIGHTED_RESULT_{fold}.pkl', 'wb') as weighted_result:
            pickle.dump(result_weighted.x, weighted_result)

        final_rmse = (result_weighted.x[0]*vit_swin_rmse +
                      result_weighted.x[1]*ridge_rmse)
        logger.info(f"Final RMSE: {final_rmse}")


def ensemble_train_stacking_with_second_level_fold():
    import scipy.optimize as optimize
    from cuml import SVR, Ridge
    from sklearn.model_selection import KFold

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
        logger.info(f"Training fold {fold}")

        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        val_df = df.loc[df["fold"] == fold].reset_index(drop=True)

        vit_train_embeds = np.zeros((len(train_df), 384))
        swin_train_embeds = np.zeros((len(train_df), 1024))

        train_dataloader = PawDataset.as_dataloader(train_df,
                                                    Augmentation(
                                                        ensemble_config).get_augmentation_by_mode('train'),
                                                    ensemble_config,
                                                    **ensemble_config.data_loader)

        for name in ensemble_config.first_level_models:
            main_config.model_name = name
            if os.path.exists(f'SVR_FOLD_{fold}_{name}.pkl'):
                logger.info(f"Skipping training {name}, not necessary")
                continue

            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            trainer = pl.Trainer(**ensemble_config.trainer)

            train_predictions = trainer.predict(
                model, train_dataloader, return_predictions=True)

            if name == 'ViTHybridSmallv2':
                vit_train_embeds = np.concatenate(
                    [pred['embeddings'] for pred in train_predictions], axis=0)
            else:
                swin_train_embeds = np.concatenate(
                    [pred['embeddings'] for pred in train_predictions], axis=0)

        classifiers = []

        for name in ensemble_config.first_level_models:
            if os.path.exists(f'SVR_FOLD_{fold}_{name}.pkl'):
                logger.info(
                    f"Skipping training SVR_FOLD_{fold}_{name}, model already exists")
                with open(f'SVR_FOLD_{fold}_{name}.pkl', 'rb') as clf:
                    clf = pickle.load(clf)
                    classifiers.append(clf)
                continue

            logger.info(f"Train SVR on {name} embeddings")

            clf = SVR(C=20.0, verbose=True)

            if name == 'ViTHybridSmallv2':
                clf.fit(vit_train_embeds.astype('float32'),
                        train_df['Pawpularity'].values.astype('int32'))
            else:
                clf.fit(swin_train_embeds.astype('float32'),
                        train_df['Pawpularity'].values.astype('int32'))

            with open(f'SVR_FOLD_{fold}_{name}.pkl', 'wb') as f:
                pickle.dump(clf, f)

            classifiers.append(clf)

        val_dataloader = PawDataset.as_dataloader(val_df,
                                                  Augmentation(
                                                      ensemble_config).get_augmentation_by_mode('tta'),
                                                  ensemble_config,
                                                  **ensemble_config.data_loader)

        vit_val_embeds = np.zeros((len(val_df), 384))
        swin_val_embeds = np.zeros((len(val_df), 1024))

        val_targets = np.zeros(
            (len(val_df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):
            main_config.model_name = name

            if name == 'ViTHybridSmallv2':
                tta_embed_preds = np.zeros(
                    (ensemble_config.tta_steps, len(val_df), 384))
            else:
                tta_embed_preds = np.zeros(
                    (ensemble_config.tta_steps, len(val_df), 1024))

            tta_preds = np.zeros((len(val_df), ensemble_config.tta_steps))
            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            for step in range(ensemble_config.tta_steps):
                trainer = pl.Trainer(**ensemble_config.trainer)

                valid_predictions = trainer.predict(
                    model, val_dataloader, return_predictions=True)

                tta_embed_preds[step, ...] = np.concatenate(
                    [pred['embeddings'] for pred in valid_predictions], axis=0)
                tta_preds[:, step] = np.concatenate(
                    [pred['pred'] for pred in valid_predictions], axis=0)

            if name == 'ViTHybridSmallv2':
                vit_val_embeds = np.apply_along_axis(
                    np.mean, axis=0, arr=tta_embed_preds)
            else:
                swin_val_embeds = np.apply_along_axis(
                    np.mean, axis=0, arr=tta_embed_preds)

            val_targets[:, idx] = np.apply_along_axis(
                np.mean, axis=1, arr=tta_preds)

        oof = np.zeros((len(val_df), len(ensemble_config.first_level_models)))

        for idx, (embeds, name, clf) in enumerate(zip([vit_val_embeds, swin_val_embeds], ensemble_config.first_level_models, classifiers)):

            clf_preds = clf.predict(embeds.astype('float32'))

            oof[:, idx] = clf_preds

            targets = val_df['Pawpularity'].values

            clf_rmse = np.sqrt(np.mean((targets - clf_preds)**2.0))

            logger.info(f"SVR fold: {fold} name: {name} RMSE: {clf_rmse}")

        targets = val_df['Pawpularity'].values

        kf = KFold(n_splits=ensemble_config.n_folds)

        holdout_preds = []

        for idx, (train_idx, val_idx) in enumerate(kf.split(oof)):
            holdout_train, holdout_train_target = oof[train_idx], targets[train_idx]
            holdout_val, holdout_val_target = oof[val_idx], targets[val_idx]
            if os.path.exists(f'RIDGE_FOLD_{fold}_{idx}.pkl'):
                logger.info(
                    f'Skipping training RIDGE_FOLD_{fold}_{idx}, model already exists')
                with open(f'RIDGE_FOLD_{fold}_{idx}.pkl', 'rb') as clf:
                    ridge_clf = pickle.load(clf)
                ridge_preds = ridge_clf.predict(holdout_val)
            else:
                ridge_clf = Ridge(alpha=1.0, solver='svd', verbose=True)
                ridge_clf.fit(holdout_train.astype('float32'),
                              holdout_train_target.astype('int32'))
                with open(f'RIDGE_FOLD_{fold}_{idx}.pkl', 'wb') as f:
                    pickle.dump(ridge_clf, f)
                ridge_preds = ridge_clf.predict(holdout_val)
            ridge_rmse = np.sqrt(
                np.mean((holdout_val_target - ridge_preds)**2.0))
            holdout_preds.append(ridge_rmse)

        vit_rmse = np.sqrt(np.mean((targets - val_targets[:, 0])**2.0))
        swin_rmse = np.sqrt(np.mean((targets - val_targets[:, 1])**2.0))
        ridge_holdout_rmse = sum(holdout_preds) / len(holdout_preds)

        logger.info(f"VIT fold: {fold} RMSE: {vit_rmse}")
        logger.info(f"Swin fold: {fold} RMSE: {swin_rmse}")
        logger.info(f"Ridge fold: {fold} RMSE: {ridge_holdout_rmse}")

        def minimize_rmse(x, *args):
            vit_preds = args[0][:, 0]
            swin_preds = args[0][:, 1]
            targets = args[1]
            oof = (1-x)*vit_preds + x*swin_preds
            oof_rmse = np.sqrt(np.mean(targets - oof) ** 2)
            return oof_rmse

        def minimize_weighted_rmse(x, *args):
            vit_swin_rmse = args[0]
            ridge_holdout_rmse = args[1]
            oof = (1-x)*vit_swin_rmse + x*ridge_holdout_rmse
            return oof

        result = optimize.differential_evolution(minimize_rmse,
                                                 args=(val_targets, targets),
                                                 bounds=((0, 1),))

        with open(f'VIT_SWIN_RESULT_{fold}.pkl', 'wb') as vit_swin_result:
            pickle.dump(result.x, vit_swin_result)

        vit_swin_ensemble = (
            1-result.x)*np.array(val_targets[:, 0]) + result.x*np.array(val_targets[:, 1])
        vit_swin_rmse = np.sqrt(np.mean((targets - vit_swin_ensemble)**2.0))
        logger.info(f"Vit/Swin Ensemble RMSE: {vit_swin_rmse}")

        result_weighted = optimize.differential_evolution(minimize_weighted_rmse,
                                                          args=(vit_swin_rmse, ridge_holdout_rmse),
                                                          bounds=((0, 1),))

        with open(f'WEIGHTED_RESULT_{fold}.pkl', 'wb') as weighted_result:
            pickle.dump(result_weighted.x, weighted_result)

        final_rmse = (1-result_weighted.x)*vit_swin_rmse + result_weighted.x*ridge_holdout_rmse
        logger.info(f"Final RMSE: {final_rmse}")


def ensemble_train_vit_swin_svr():
    import scipy.optimize as optimize
    from cuml import SVR

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
        logger.info(f"Training fold {fold}")

        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        val_df = df.loc[df["fold"] == fold].reset_index(drop=True)

        vit_train_embeds = np.zeros((len(train_df), 384))
        swin_train_embeds = np.zeros((len(train_df), 1024))

        train_dataloader = PawDataset.as_dataloader(train_df,
                                                    Augmentation(
                                                        ensemble_config).get_augmentation_by_mode('train'),
                                                    ensemble_config,
                                                    **ensemble_config.data_loader)

        for name in ensemble_config.first_level_models:
            main_config.model_name = name
            if os.path.exists(f'SVR_FOLD_{fold}_{name}.pkl'):
                logger.info(f"Skipping training {name}, not necessary")
                continue

            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            trainer = pl.Trainer(**ensemble_config.trainer)

            train_predictions = trainer.predict(
                model, train_dataloader, return_predictions=True)

            if name == 'ViTHybridSmallv2':
                vit_train_embeds = np.concatenate(
                    [pred['embeddings'] for pred in train_predictions], axis=0)
            else:
                swin_train_embeds = np.concatenate(
                    [pred['embeddings'] for pred in train_predictions], axis=0)

        classifiers = []

        for name in ensemble_config.first_level_models:
            if os.path.exists(f'SVR_FOLD_{fold}_{name}.pkl'):
                logger.info(
                    f"Skipping training SVR_FOLD_{fold}_{name}, model already exists")
                with open(f'SVR_FOLD_{fold}_{name}.pkl', 'rb') as clf:
                    clf = pickle.load(clf)
                    classifiers.append(clf)
                continue

            logger.info(f"Train SVR on {name} embeddings")

            clf = SVR(C=20.0, verbose=True)

            if name == 'ViTHybridSmallv2':
                clf.fit(vit_train_embeds.astype('float32'),
                        train_df['Pawpularity'].values.astype('int32'))
            else:
                clf.fit(swin_train_embeds.astype('float32'),
                        train_df['Pawpularity'].values.astype('int32'))

            with open(f'SVR_FOLD_{fold}_{name}.pkl', 'wb') as f:
                pickle.dump(clf, f)

            classifiers.append(clf)

        val_dataloader = PawDataset.as_dataloader(val_df,
                                                  Augmentation(
                                                      ensemble_config).get_augmentation_by_mode('tta'),
                                                  ensemble_config,
                                                  **ensemble_config.data_loader)

        vit_val_embeds = np.zeros((len(val_df), 384))
        swin_val_embeds = np.zeros((len(val_df), 1024))

        val_targets = np.zeros(
            (len(val_df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):
            main_config.model_name = name

            if name == 'ViTHybridSmallv2':
                tta_embed_preds = np.zeros(
                    (ensemble_config.tta_steps, len(val_df), 384))
            else:
                tta_embed_preds = np.zeros(
                    (ensemble_config.tta_steps, len(val_df), 1024))

            tta_preds = np.zeros((len(val_df), ensemble_config.tta_steps))
            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=main_config)

            for step in range(ensemble_config.tta_steps):
                trainer = pl.Trainer(**ensemble_config.trainer)

                valid_predictions = trainer.predict(
                    model, val_dataloader, return_predictions=True)

                tta_embed_preds[step, ...] = np.concatenate(
                    [pred['embeddings'] for pred in valid_predictions], axis=0)
                tta_preds[:, step] = np.concatenate(
                    [pred['pred'] for pred in valid_predictions], axis=0)

            if name == 'ViTHybridSmallv2':
                vit_val_embeds = np.apply_along_axis(
                    np.mean, axis=0, arr=tta_embed_preds)
            else:
                swin_val_embeds = np.apply_along_axis(
                    np.mean, axis=0, arr=tta_embed_preds)

            val_targets[:, idx] = np.apply_along_axis(
                np.mean, axis=1, arr=tta_preds)

        oof = np.zeros((len(val_df), len(ensemble_config.first_level_models)))

        for idx, (embeds, name, clf) in enumerate(zip([vit_val_embeds, swin_val_embeds], ensemble_config.first_level_models, classifiers)):

            clf_preds = clf.predict(embeds.astype('float32'))

            oof[:, idx] = clf_preds

            targets = val_df['Pawpularity'].values

            clf_rmse = np.sqrt(np.mean((targets - clf_preds)**2.0))

            logger.info(f"SVR fold: {fold} name: {name} RMSE: {clf_rmse}")

        targets = val_df['Pawpularity'].values

        vit_rmse = np.sqrt(np.mean((targets - val_targets[:, 0])**2.0))
        swin_rmse = np.sqrt(np.mean((targets - val_targets[:, 1])**2.0))

        logger.info(f"VIT fold: {fold} RMSE: {vit_rmse}")
        logger.info(f"Swin fold: {fold} RMSE: {swin_rmse}")

        def minimize_weighted_rmse(x, *args):
            vit_preds = args[0][:, 0]
            swin_preds = args[0][:, 1]
            vit_svr_preds = args[1][:, 0]
            swin_svr_preds = args[1][:, 1]
            targets = args[2]
            final_preds = x[0]*vit_preds+x[1]*swin_preds + \
                x[2]*vit_svr_preds+x[3]*swin_svr_preds
            final_rmse = np.sqrt(np.mean(targets - final_preds) ** 2)
            return final_rmse

        def fconstr(x): return 1 - sum(x)

        constraint = optimize.NonlinearConstraint(fconstr, 0, 1)
        result_weighted = optimize.differential_evolution(minimize_weighted_rmse,
                                                          args=(
                                                              val_targets, oof, targets),
                                                          bounds=(
                                                              (0, 1), (0, 1), (0, 1), (0, 1)),
                                                          constraints=(constraint))

        with open(f'WEIGHTED_RESULT_{fold}.pkl', 'wb') as weighted_result:
            pickle.dump(result_weighted.x, weighted_result)

        final_oof = result_weighted.x[0]*val_targets[:, 0] + result_weighted.x[1] * \
            val_targets[:, 1] + result_weighted.x[2] * \
            oof[:, 0] + result_weighted.x[3] * oof[:, 1]
        
        final_rmse = np.sqrt(np.mean((targets - final_oof)**2.0))

        logger.info(f"Final RMSE: {final_rmse}")
