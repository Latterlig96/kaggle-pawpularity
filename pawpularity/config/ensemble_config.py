import dataclasses
import typing


@dataclasses.dataclass
class EnsembleConfig:
    n_folds: int = 10
    first_level_models: typing.List[str] = ['VitHybridSmallv2', 'SwinSmall']
    second_level_models: typing.List[str] = ['SVR',]
    tta_steps: int = 5
    image_size: typing.Tuple[int] = (768, 768)
    image_mean: typing.Tuple[float] = (0.485, 0.456, 0.406)
    image_std: typing.Tuple[float] = (0.229, 0.224, 0.225)
    
    data_loader: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(default_factory=lambda: {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 6,
        'pin_memory': False,
        'drop_last': False
    })

    @property
    def asdict(self):
        return dataclasses.asdict(self)
