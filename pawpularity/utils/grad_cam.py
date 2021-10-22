from typing import TypeVar
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.model_selection import StratifiedKFold
from pawpularity.augmentations import Augmentation
from pawpularity.config import Config
from pawpularity.datasets import PawModule
from pawpularity.models import Model

__all__ = ('show_cam', )

Tensor = TypeVar("Tensor")

def _reshape_transform(tensor: Tensor, height: int = 7, width: int = 7):
    result = tensor.reshape(tensor.size(0),
                            height, width, 
                            tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result 

def _yield_indices(df, config):
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        yield train_idx, val_idx

def show_cam():
    config = Config()

    model = Model(config)
    model.load_from_checkpoint(f'{config.model_name}/default/version_0/checkpoints/best_loss.ckpt', cfg=config)
    model = model.eval().cuda()

    df_path = config.root + '/' + 'train.csv'
    img_path = config.root + '/' 'train'
    df = pd.read_csv(df_path)
    df["Id"] = df["Id"].apply(lambda x: img_path + '/' + x + '.jpg')
    train_idx, val_idx = next(_yield_indices(df, config))

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)

    config.val_loader['batch_size'] = 9
    datamodule = PawModule(train_df, val_df, Augmentation, config)
    images, grayscale_cams, preds, labels = model.check_gradcam(
        dataloader=datamodule.val_dataloader(),
        target_layer=model.model.backbone.layers[-1].blocks[-1].norm1,
        target_category=None,
        reshape_transform=_reshape_transform)
    
    plt.figure(figsize=(12, 12))
    for it, (image, grayscale_cam, pred, label) in enumerate(zip(images, grayscale_cams, preds, labels)):
        plt.subplot(4, 4, it + 1)
        visualization = show_cam_on_image(image, grayscale_cam)
        plt.imshow(visualization)
        plt.title(f'pred: {pred:.1f} label: {label}')
        plt.axis('off')
    plt.show()
