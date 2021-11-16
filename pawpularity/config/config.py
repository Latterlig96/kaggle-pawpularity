import dataclasses
import typing


@dataclasses.dataclass
class Config:
    seed: int = 3407
    epochs: int = 5
    shuffle: bool = True
    n_splits: int = 5
    root: str = './pawpularity/data'
    model_name: str = 'ViTHybridSmallv2'
    use_pretrained: bool = True
    image_size: typing.Tuple[int] = (768, 768)
    image_mean: typing.Tuple[float] = (0.485, 0.456, 0.406)
    image_std: typing.Tuple[float] = (0.229, 0.224, 0.225)
    output_dim: int = 1
    use_dropout: bool = True
    dropout_rate: float = 0.2
    verbose: bool = True
    patience: int = 3

    trainer: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'gpus': 1,
        'accumulate_grad_batches': 1,
        'progress_bar_refresh_rate': 1,
        'fast_dev_run': False,
        'num_sanity_val_steps': 0,
        'resume_from_checkpoint': None,
    })

    resizer: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'apply': True,
        'in_channels': 3,
        'out_channels': 3,
        'num_kernels': 16,
        'num_resblocks': 2,
        'negative_slope': 0.2,
        'interpolation_mode': 'bilinear',
        'target_size': [384, 384]
    })

    stn: typing.Dict[str, typing.Union[str, bool]] = dataclasses.field(default_factory=lambda: {
        'apply': True,
        'apply_after_resizer': False
    })

    augmentation: typing.Dict[str, typing.Union[str, float]] = dataclasses.field(default_factory=lambda: {
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

    })

    train_loader: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 6,
        'pin_memory': False,
        'drop_last': True
    })

    val_loader: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 6,
        'pin_memory': False,
        'drop_last': False
    })

    optimizer: typing.Dict[str, typing.Union[str, float]] = dataclasses.field(default_factory=lambda: {
        'name': "AdamW",
        'params': {
            'lr': 1e-5
        }
    })

    scheduler: typing.Dict[str, typing.Union[str, float]] = dataclasses.field(default_factory=lambda: {
        'name': 'CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 20,
            'eta_min': 1e-4,
        }
    })

    loss: str = 'BCEWithLogitsLoss'

    @property
    def asdict(self):
        return dataclasses.asdict(self)
