import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(
    description="Make fold to use later in training")

parser.add_argument('-d','--df',
                    help="DataFrame path",
                    dest='df',
                    default=None)

parser.add_argument('-f','--folds',
                    help="Number of folds to create",
                    dest='folds',
                    default=None,
                    type=int)

parser.add_argument('-s','--seed',
                    help="random seed",
                    dest='seed',
                    default=None,
                    type=int)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.df is None:
        raise ValueError("Dataframe not provided, please try again")
    df = pd.read_csv(args.df)

    df["fold"] = -1

    skf = StratifiedKFold(n_splits=args.folds,
                          shuffle=True, random_state=args.seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        df.loc[val_idx, "fold"] = fold
    
    df.to_csv(f'pawpularity_data_{args.folds}_folds_{args.seed}_seed.csv')
