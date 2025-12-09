import pandas as pd

from preprocessing.create_lookup_table import create_lookup_table
from preprocessing.create_splits_baseline import write_data_to_dir, create_train_test_val_splits
from preprocessing.create_splits_no_traj_overlap import create_train_test_val_splits_traj_overlap, write_data_to_dir_traj_overlap

if __name__ == "__main__":
    create_lookup_table().to_csv('../data/fish_lookup_table.csv')
    df = pd.read_csv('../data/fish_lookup_table.csv')
    tr, te, val = create_train_test_val_splits(df)
    write_data_to_dir(tr, te, val)

    tr, te, val = create_train_test_val_splits_traj_overlap(df)
    write_data_to_dir_traj_overlap(tr, te, val)