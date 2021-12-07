import argparse
from .test import test_ensemble_stacking_with_second_level_fold, test_ensemble_stacking_without_second_level_fold, \
    test_ensemble_vit_swin_svr, test_main
from .train import ensemble_train_stacking_with_second_level_fold, ensemble_train_stacking_without_second_level_fold, \
    ensemble_train_vit_swin_svr, resizer_train_main, train_main
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
    elif args.mode == "ensemble-train-stacking-wihout-second-level-fold":
        ensemble_train_stacking_without_second_level_fold()
    elif args.mode == "ensemble-train-stacking-with-second-level-fold":
        ensemble_train_stacking_with_second_level_fold()
    elif args.mode == "ensemble-train-vit-swin-svr":
        ensemble_train_vit_swin_svr()
    elif args.mode == "test":
        test_main()
    elif args.mode == "test-ensemble-stacking-with_second_level_fold":
        test_ensemble_stacking_with_second_level_fold()
    elif args.mode == "test-ensemble_train_stacking_without_second_level_fold":
        test_ensemble_stacking_without_second_level_fold()
    elif args.mode == "test-ensemble-vit-swin-svr":
        test_ensemble_vit_swin_svr()
    elif args.mode == "utils":
        show_cam()
        show_training_results()
