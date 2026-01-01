import os
import pandas as pd

from preprocessing.create_lookup_table import create_lookup_table
from preprocessing.create_splits_baseline import write_data_to_dir, create_train_test_val_splits
from preprocessing.create_splits_no_traj_overlap import create_train_test_val_splits_traj_overlap, write_data_to_dir_traj_overlap


def main():
    df = create_lookup_table()


    tr, te, val, new = create_train_test_val_splits(df)
    df.loc[:,'base_split'] = new.loc[:, 'base_split']
    write_data_to_dir(tr, te, val)

    tr, te, val, new = create_train_test_val_splits_traj_overlap(df)
    write_data_to_dir_traj_overlap(tr, te, val)
    df.loc[:,'traj_split'] = new.loc[:, 'traj_split']

    df.to_csv('lookup_table.csv')


if __name__ == "__main__":
    main()