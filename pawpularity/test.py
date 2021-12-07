import os
import pickle
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from pawpularity.augmentations import Augmentation, ValAugmentation
from pawpularity.config import Config, EnsembleConfig
from pawpularity.datasets import PawDataset
from pawpularity.models import Model


@torch.no_grad()
def test_main():
    config = Config()
    df_path = config.root_df
    img_path = config.root_img
    submission_path = config.root_submission
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_model_dirs = os.path.join(os.getcwd(), config.model_name, 'default')

    model = Model(config)
    stored_predictions = np.zeros((len(df), config.n_splits))

    for index, version in enumerate(os.listdir(root_model_dirs)):
        model = model.load_from_checkpoint(os.path.join(
            root_model_dirs, version, 'checkpoints', 'best_loss.ckpt'), cfg=config)
        model = model.eval().cuda()
        predictions = np.zeros(len(df))
        for idx, image in enumerate(df["Id"].tolist()):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            image = ValAugmentation(config)(image).unsqueeze(0).to(device)
            prediction = model(image)
            predictions[idx] = prediction.sigmoid().detach().cpu() * 100.
        stored_predictions[:, index] = predictions

    stored_predictions = np.apply_along_axis(
        np.mean, axis=1, arr=stored_predictions)

    df['Id'] = df['Id'].apply(lambda x: x.replace(config.root, '')).\
        apply(lambda x: x.replace('/', '')).\
        apply(lambda x: x.replace('.jpg', '')).\
        apply(lambda x: x.replace('test', ''))

    pred_df = pd.DataFrame({'Id': df.Id,
                            'Pawpularity': stored_predictions.reshape(-1)})
    sub_df = pd.read_csv(submission_path)
    del sub_df['Pawpularity']
    sub_df = sub_df.merge(pred_df, on='Id', how='left')
    sub_df.to_csv('submission.csv', index=False)


def test_ensemble_stacking_without_second_level_fold():
    config = Config()
    ensemble_config = EnsembleConfig()

    df_path = config.root_df
    img_path = config.root_img
    submission_path = config.root_submission
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    folds = ensemble_config.n_folds

    final_preds = np.zeros((len(df), len(folds)))

    for fold in range(folds):
        logger.info(f"Inference on fold {fold}")

        test_embeddings = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))
        test_logits = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):
            config.model_name = name
            logger.info(f"Inference on fold {fold} with {name}")

            tta_logits = np.zeros((len(df), len(ensemble_config.tta_steps)))
            tta_embeddings = np.zeros(
                (len(df), len(ensemble_config.tta_steps)))

            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=config)

            for step in range(ensemble_config.tta_steps):
                test_dataloader = PawDataset.as_dataloader(df,
                                                           Augmentation.get_augmentation_by_mode(
                                                               'tta'),
                                                           ensemble_config)

                trainer = pl.Trainer(**ensemble_config.trainer)

                test_predictions = trainer.predict(
                    model, test_dataloader, return_predictions=True)

                tta_embeddings[:, step] = np.concatenate(
                    [pred['embeddings'] for pred in test_predictions], axis=0)
                tta_logits[:, step] = np.concatenate(
                    [pred['pred'] for pred in test_predictions], axis=0)

            test_embeddings[:, idx] = np.apply_along_axis(
                np.mean, 1, tta_embeddings)
            test_logits[:, idx] = np.apply_along_axis(np.mean, 1, tta_logits)

        test_svr_preds = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))
        for idx, name in enumerate(ensemble_config.first_level_models):
            logger.info(f"Inference on fold {fold} with {name} SVR")
            with open(f'SVR_FOLD_{fold}_{name}.pkl', 'rb') as svr_clf:
                svr_clf = pickle.load(svr_clf)
                test_svr_preds[:, idx] = svr_clf.predict(
                    test_embeddings[:, idx].astype('float32'))

        with open(f'RIDGE_FOLD_{fold}.pkl', 'rb') as ridge_clf:
            ridge_clf = pickle.load(ridge_clf)
            ridge_preds = ridge_clf.predict(test_svr_preds)

        with open(f'VIT_SWIN_RESULT_{fold}.pkl', 'rb') as vit_swin_weight:
            weight = pickle.load(vit_swin_weight)

        vit_swin_ensemble = (1 - weight) * \
            test_logits[:, 0] + weight*test_logits[:, 1]

        with open(f'WEIGHTED_RESULT_{fold}.pkl', 'rb') as final_weight:
            final_weight = pickle.load(final_weight)

        final_oof_preds = (1-final_weight) * \
            vit_swin_ensemble + final_weight*ridge_preds

        final_preds[:, fold] = final_oof_preds

    final_preds = np.apply_along_axis(np.mean, 1, final_preds)

    df['Id'] = df['Id'].apply(lambda x: x.replace(config.root, '')).\
        apply(lambda x: x.replace('/', '')).\
        apply(lambda x: x.replace('.jpg', '')).\
        apply(lambda x: x.replace('test', ''))

    pred_df = pd.DataFrame({'Id': df.Id,
                            'Pawpularity': final_preds.reshape(-1)})
    sub_df = pd.read_csv(submission_path)
    del sub_df['Pawpularity']
    sub_df = sub_df.merge(pred_df, on='Id', how='left')
    sub_df.to_csv('submission.csv', index=False)


def test_ensemble_stacking_with_second_level_fold():
    config = Config()
    ensemble_config = EnsembleConfig()

    df_path = config.root_df
    img_path = config.root_img
    submission_path = config.root_submission
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    folds = ensemble_config.n_folds

    final_preds = np.zeros((len(df), len(folds)))

    for fold in range(folds):
        logger.info(f"Inference on fold {fold}")

        test_embeddings = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))
        test_logits = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):
            config.model_name = name
            logger.info(f"Inference on fold {fold} with {name}")

            tta_logits = np.zeros((len(df), len(ensemble_config.tta_steps)))
            tta_embeddings = np.zeros(
                (len(df), len(ensemble_config.tta_steps)))

            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=config)

            for step in range(ensemble_config.tta_steps):
                test_dataloader = PawDataset.as_dataloader(df,
                                                           Augmentation.get_augmentation_by_mode(
                                                               'tta'),
                                                           ensemble_config)

                trainer = pl.Trainer(**ensemble_config.trainer)

                test_predictions = trainer.predict(
                    model, test_dataloader, return_predictions=True)

                tta_embeddings[:, step] = np.concatenate(
                    [pred['embeddings'] for pred in test_predictions], axis=0)
                tta_logits[:, step] = np.concatenate(
                    [pred['pred'] for pred in test_predictions], axis=0)

            test_embeddings[:, idx] = np.apply_along_axis(
                np.mean, 1, tta_embeddings)
            test_logits[:, idx] = np.apply_along_axis(np.mean, 1, tta_logits)

        test_svr_preds = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))
        for idx, name in enumerate(ensemble_config.first_level_models):
            logger.info(f"Inference on fold {fold} with {name} SVR")
            with open(f'SVR_FOLD_{fold}_{name}.pkl', 'rb') as svr_clf:
                svr_clf = pickle.load(svr_clf)
                test_svr_preds[:, idx] = svr_clf.predict(
                    test_embeddings[:, idx].astype('float32'))

        ridge_oof_preds = np.zeros((len(df), len(folds)))

        for idx in range(folds):
            with open(f'RIDGE_FOLD_{fold}_{idx}.pkl', 'rb') as ridge_clf:
                ridge_clf = pickle.load(ridge_clf)
                ridge_oof_preds[:, idx] = ridge_clf.predict(
                    test_svr_preds.astype('float32'))

        ridge_oof_preds = np.apply_along_axis(np.mean, 1, ridge_oof_preds)

        with open(f'VIT_SWIN_RESULT_{fold}.pkl', 'rb') as weight:
            weight = pickle.load(weight)

        vit_swin_ensemble = (1 - weight) * \
            test_logits[:, 0] + weight*test_logits[:, 1]

        with open(f'WEIGHTED_RESULT_{fold}.pkl', 'rb') as final_weight:
            final_weight = pickle.load(final_weight)

        final_oof_preds = (1-final_weight)*vit_swin_ensemble + \
            final_weight*ridge_oof_preds

        final_preds[:, fold] = final_oof_preds

    final_preds = np.apply_along_axis(np.mean, 1, final_preds)

    df['Id'] = df['Id'].apply(lambda x: x.replace(config.root, '')).\
        apply(lambda x: x.replace('/', '')).\
        apply(lambda x: x.replace('.jpg', '')).\
        apply(lambda x: x.replace('test', ''))

    pred_df = pd.DataFrame({'Id': df.Id,
                            'Pawpularity': final_preds.reshape(-1)})
    sub_df = pd.read_csv(submission_path)
    del sub_df['Pawpularity']
    sub_df = sub_df.merge(pred_df, on='Id', how='left')
    sub_df.to_csv('submission.csv', index=False)


def test_ensemble_vit_swin_svr():
    config = Config()
    ensemble_config = EnsembleConfig()

    df_path = config.root_df
    img_path = config.root_img
    submission_path = config.root_submission
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')

    folds = ensemble_config.n_folds

    final_preds = np.zeros((len(df), len(folds)))

    for fold in range(folds):
        logger.info(f"Inference on fold {fold}")

        test_embeddings = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))
        test_logits = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))

        for idx, name in enumerate(ensemble_config.first_level_models):
            config.model_name = name
            logger.info(f"Inference on fold {fold} with {name}")

            tta_logits = np.zeros((len(df), len(ensemble_config.tta_steps)))
            tta_embeddings = np.zeros(
                (len(df), len(ensemble_config.tta_steps)))

            model = Model.load_from_checkpoint(os.path.join(
                name, 'default', f'version_{fold}', 'checkpoints', 'best_loss.ckpt'), cfg=config)

            for step in range(ensemble_config.tta_steps):
                test_dataloader = PawDataset.as_dataloader(df,
                                                           Augmentation.get_augmentation_by_mode(
                                                               'tta'),
                                                           ensemble_config)

                trainer = pl.Trainer(**ensemble_config.trainer)

                test_predictions = trainer.predict(
                    model, test_dataloader, return_predictions=True)

                tta_embeddings[:, step] = np.concatenate(
                    [pred['embeddings'] for pred in test_predictions], axis=0)
                tta_logits[:, step] = np.concatenate(
                    [pred['pred'] for pred in test_predictions], axis=0)

            test_embeddings[:, idx] = np.apply_along_axis(
                np.mean, 1, tta_embeddings)
            test_logits[:, idx] = np.apply_along_axis(np.mean, 1, tta_logits)

        test_svr_preds = np.zeros(
            (len(df), len(ensemble_config.first_level_models)))
        for idx, name in enumerate(ensemble_config.first_level_models):
            logger.info(f"Inference on fold {fold} with {name} SVR")
            with open(f'SVR_FOLD_{fold}_{name}.pkl', 'rb') as svr_clf:
                svr_clf = pickle.load(svr_clf)
                test_svr_preds[:, idx] = svr_clf.predict(
                    test_embeddings[:, idx].astype('float32'))

        with open(f'WEIGHTED_RESULT_{fold}.pkl', 'rb') as final_weights:
            final_weights = pickle.load(final_weights)

        final_oof_preds = final_weights[0] * test_logits[:, 0] + final_weights[1] * \
            test_logits[:, 1] + final_weights[2] * test_svr_preds[:, 0] + final_weights[3] * \
            test_svr_preds[:, 1]

        final_preds[:, fold] = final_oof_preds

    final_preds = np.apply_along_axis(np.mean, 1, final_preds)

    df['Id'] = df['Id'].apply(lambda x: x.replace(config.root, '')).\
        apply(lambda x: x.replace('/', '')).\
        apply(lambda x: x.replace('.jpg', '')).\
        apply(lambda x: x.replace('test', ''))

    pred_df = pd.DataFrame({'Id': df.Id,
                            'Pawpularity': final_preds.reshape(-1)})
    sub_df = pd.read_csv(submission_path)
    del sub_df['Pawpularity']
    sub_df = sub_df.merge(pred_df, on='Id', how='left')
    sub_df.to_csv('submission.csv', index=False)
