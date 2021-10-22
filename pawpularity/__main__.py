import argparse
from .train import train_main
from .test import test_main
from .utils import show_cam, show_training_results

parser = argparse.ArgumentParser(description="Kaggle Pawpularity Competition")

parser.add_argument('-m',
                    '--mode',
                    dest='mode',
                    default="train",
                    help="Mode - whether to run in training mode or test mode",
                    type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.mode == "train":
        train_main()
    elif args.mode == "test":
        test_main()
    elif args.mode == "utils":
        show_cam()
        show_training_results()
