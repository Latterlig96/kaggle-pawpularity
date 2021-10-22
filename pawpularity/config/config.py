class Config:
    seed: int = 47
    epochs: int = 100
    shuffle: bool = True
    n_splits: int = 5
    root: str = './pawpularity/data'
    model_name: str = 'SwinSmall'
    use_pretrained: bool = False
    image_size: list = [224, 224]
    image_mean: list = [0.485, 0.456, 0.406]
    image_std: list = [0.229, 0.224, 0.225]
    output_dim: int = 1

    trainer: dict = {
              'gpus': 1,
              'accumulate_grad_batches': 1,
              'progress_bar_refresh_rate': 1,
              'fast_dev_run': False,
              'num_sanity_val_steps': 0,
              'resume_from_checkpoint': None,
        }
    
    augmentation: dict = {
        'color_jitter': {
            'brightness': 0.1,
            'contrast': 0.1,
            'saturation': 0.1,
            'hue': 0.1
        },
        'affine': {
            'scale': (0.9, 1.1),
            'translate_percent': (0.1, 0.1),
            'rotate': 15
        }

    }

    train_loader: dict = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 6,
        'pin_memory': False,
        'drop_last': True
        }

    val_loader: dict = {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 6,
        'pin_memory': False,
        'drop_last': False
        }

    optimizer: dict = {
        'name': "Adam",
        'params': {
            'lr': 1e-5
        }
        }

    scheduler: dict = {
              'name': 'CosineAnnealingWarmRestarts',
              'params':{
                  'T_0': 20,
                  'eta_min': 1e-4,
              }
        }

    loss: str = 'BCEWithLogitsLoss'
