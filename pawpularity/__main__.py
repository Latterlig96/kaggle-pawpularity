import argparse
from .train import train_main, resizer_train_main, \
                   ensemble_train_stacking_wihout_second_level_fold, \
                   ensemble_train_stacking_with_second_level_fold, \
                   ensemble_train_vit_swin_svr
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
    elif args.mode == "train-resizer":
        resizer_train_main()
    elif args.mode == "ensemble_train_stacking_wihout_second_level_fold":
        ensemble_train_stacking_wihout_second_level_fold()
    elif args.mode == "ensemble_train_stacking_with_second_level_fold":
        ensemble_train_stacking_with_second_level_fold()
    elif args.mode == "ensemble_train_vit_swin_svr":
        ensemble_train_vit_swin_svr()
    elif args.mode == "test":
        test_main()
    elif args.mode == "utils":
        show_cam()
        show_training_results()
