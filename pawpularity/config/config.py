import dataclasses
import typing
import yaml


@dataclasses.dataclass
class Config:
    seed: int = 3407
    epochs: int = 5
    shuffle: bool = True
    n_splits: int = 10
    root_df: str = './pawpularity/data/pawpularity_data_10_folds_999_seed.csv'
    root_img: str = './pawpularity/data/train'
    root_submission: str = './pawpularity/data/sample_submission.csv'
    model_name: str = 'ViTHybridSmallv2'
    use_pretrained: bool = True
    image_size: typing.Tuple[int] = (384, 384)
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
        'apply': False,
        'in_channels': 3,
        'out_channels': 3,
        'num_kernels': 16,
        'num_resblocks': 2,
        'negative_slope': 0.2,
        'interpolation_mode': 'bilinear',
        'target_size': [384, 384]
    })

    stn: typing.Dict[str, typing.Union[str, bool]] = dataclasses.field(default_factory=lambda: {
        'apply': False,
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

    def __new__(cls, *args, **kwargs):
        init_args, additional_args = {}, {}
        for name, value in kwargs.items():
            if name in cls.__annotations__: 
                init_args[name] = value
            else: 
                additional_args[name] = value
        
        new_cls = super().__new__(cls) 
        new_cls.__init__(**init_args)

        for key, value in additional_args.items():
            setattr(new_cls, key, value)

        return new_cls
    
    @classmethod
    def load_config_class(cls, config_path: str):
        if not isinstance(config_path, str):
            raise TypeError(
                  f"You must provide a config file with manually tuned configuration parameters, but got {config_path} instead")
        with open(config_path, 'rb') as cfg: 
            config = yaml.safe_load(cfg)
        cfg_dict = {}
        for key in config.keys():
            cfg_dict.update(**config[key])
        config_class = cls.__new__(cls, **cfg_dict)
        return config_class
    