import pandas as pd
import numpy as np
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

    num_bins = int(np.floor(1+(3.3)*(np.log2(len(df)))))

    df['norm_score'] = df['Pawpularity']/100

    df['bins'] = pd.cut(df['norm_score'], bins=num_bins, labels=False)

    df = df.sample(frac=1).reset_index(drop=True)

    df["fold"] = -1

    skf = StratifiedKFold(n_splits=args.folds,
                          shuffle=True, random_state=args.seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df.index, df["Pawpularity"])):
        df.loc[val_idx, "fold"] = fold
    
    df.to_csv(f'pawpularity_data_{args.folds}_folds_bins_{num_bins}_{args.seed}_seed.csv')
