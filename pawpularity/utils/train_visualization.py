from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob
from pawpularity.config import Config
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ('show_training_results', )

def show_training_results():
    config = Config()

    path = glob(f"./{config.model_name}/default/version_0/events*")[0]
    event_acc = EventAccumulator(path, size_guidance={'scalars': 0})
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]

    sns.set()

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(scalars['lr-Adam'])), scalars['lr-Adam'])
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('Adam lr')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(scalars['train_loss'])), scalars['train_loss'], label='train_loss')
    plt.plot(range(len(scalars['val_loss'])), scalars['val_loss'], label='val_loss')
    plt.legend()
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.title('train/val rmse')
    plt.show()

    print(f"BEST LOSS: {min(scalars['val_loss'])}")
