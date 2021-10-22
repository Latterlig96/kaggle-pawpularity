import pandas as pd
import torch
from pawpularity.config import Config
import numpy as np
import cv2
from pawpularity.augmentations import ValAugmentation
from pawpularity.models import Model
import os


@torch.no_grad()
def test_main():
    config = Config()
    df_path = config.root + '/' + 'test.csv'
    img_path = config.root + '/' + 'test'
    submission_path = config.root + '/' + 'sample_submission.csv'
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_model_dirs = os.path.join(os.getcwd(), config.model_name, 'default')

    model = Model(config)
    stored_predictions = np.zeros((len(df), config.n_splits))

    for index, version in enumerate(os.listdir(root_model_dirs)):
        model = model.load_from_checkpoint(os.path.join(root_model_dirs, version, 'checkpoints', 'best_loss.ckpt'), cfg=config)
        model = model.eval().cuda()
        predictions = np.zeros(len(df))
        for idx, image in enumerate(df["Id"].tolist()):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            image = ValAugmentation(config)(image).unsqueeze(0).to(device)
            prediction = model(image)
            predictions[idx] = prediction.sigmoid().detach().cpu() * 100.
        stored_predictions[:, index] = predictions
    
    stored_predictions = np.apply_along_axis(np.mean, axis=1, arr=stored_predictions)

    df['Id'] = df['Id'].apply(lambda x : x.replace(config.root, '')).\
                        apply(lambda x : x.replace('/', '')).\
                        apply(lambda x : x.replace('.jpg', '')).\
                        apply(lambda x : x.replace('test', ''))
                        
    pred_df = pd.DataFrame({'Id':df.Id,
                            'Pawpularity':stored_predictions.reshape(-1)})
    sub_df = pd.read_csv(submission_path)
    del sub_df['Pawpularity']
    sub_df = sub_df.merge(pred_df, on='Id', how='left')
    sub_df.to_csv('submission.csv',index=False)
