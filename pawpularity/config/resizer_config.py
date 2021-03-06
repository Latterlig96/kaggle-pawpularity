import dataclasses
import typing


class ResizerConfig:
    seed: int = 3407
    epochs: int = 100
    shuffle: bool = True
    root_df: str = './pawpularity/data/'
    root_img: str = './pawpularity/data/train'
    root_submission: str = './pawpularity/data/sample_submission.csv'
    input_image_size: typing.Tuple[int] = (768, 768)
    epochs: int = 100
    model_name: str = 'Resizer'
    image_mean: typing.Tuple[float] = (0.485, 0.456, 0.406)
    image_std: typing.Tuple[float] = (0.229, 0.224, 0.225)
    verbose: bool = True
    patience: int = 20
    val_size: float = 0.2

    train_loader: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 6,
        'pin_memory': False,
        'drop_last': True
    })

    val_loader: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 6,
        'pin_memory': False,
        'drop_last': False
    })

    resizer: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'in_channels': 3,
        'out_channels': 3,
        'num_kernels': 16,
        'num_resblocks': 2,
        'negative_slope': 0.2,
        'interpolation_mode': 'bilinear',
        'target_size': [384, 384]
    })

    trainer: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'gpus': 1,
        'accumulate_grad_batches': 1,
        'progress_bar_refresh_rate': 1,
        'fast_dev_run': False,
        'num_sanity_val_steps': 0,
        'resume_from_checkpoint': None,
    })

    optimizer: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'name': "Adam",
        'params': {
            'lr': 1e-5
        }
    })

    scheduler: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'name': 'CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 20,
            'eta_min': 1e-4,
        }
    })

    loss: str = 'MSE'
