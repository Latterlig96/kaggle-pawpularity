class Config:
    seed: int = 47
    epochs: int = 100
    n_splits: int = 5
    image_size: list = [224, 224]
    image_mean: list = [0.485, 0.456, 0.406]
    image_std: list = [0.229, 0.224, 0.225]
    trainer: dict = {
              'gpus': 1,
              'accumulate_grad_batches': 1,
              'progress_bar_refresh_rate': 1,
              'fast_dev_run': False,
              'num_sanity_val_steps': 0,
              'resume_from_checkpoint': None,
        },
    train_loader: dict = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': False,
        'drop_last': True
        }
    val_loader: dict = {
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': False,
        'drop_last': False
        }
    optimizer: dict = {
        'name': "AdamW",
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
        },
    loss: dict = 'BCEWithLogitsLoss'
