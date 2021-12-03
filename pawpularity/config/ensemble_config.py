import dataclasses
import typing


@dataclasses.dataclass
class EnsembleConfig:
    n_folds: int = 10
    root_df: str = './pawpularity/data/pawpularity_data_10_folds_999_seed.csv'
    root_img: str = './pawpularity/data/train'
    first_level_models: typing.Tuple[str] = ('ViTHybridSmallv2', 'SwinLargev2')
    second_level_models: typing.Tuple[str] = ('SVR',)
    third_level_models: typing.Tuple[str] = ('Ridge',)
    holdout_percent: float = 0.2
    tta_steps: int = 5
    image_size: typing.Tuple[int] = (384, 384)
    image_mean: typing.Tuple[float] = (0.485, 0.456, 0.406)
    image_std: typing.Tuple[float] = (0.229, 0.224, 0.225)
    
    data_loader: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda : {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 2,
        'pin_memory': False,
        'drop_last': False
    })

    trainer: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'gpus': 1,
        'accumulate_grad_batches': 1,
        'progress_bar_refresh_rate': 1,
        'fast_dev_run': False,
        'num_sanity_val_steps': 0,
        'resume_from_checkpoint': None,
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

    @property
    def asdict(self):
        return dataclasses.asdict(self)
