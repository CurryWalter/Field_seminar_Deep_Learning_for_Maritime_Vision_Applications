import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_val_splits(df, train_ratio=4/7, validation_ratio=1/7, test_ratio=2/7):
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, 'relative_path'], df.loc[:, 'label'],
                                                        test_size=1 - train_ratio,
                                                        stratify=df.loc[:, 'label'], random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio/(test_ratio + validation_ratio),
                                                    stratify=y_test.loc[:], random_state=0)


    df_train = pd.concat([x_train, y_train], axis=1)
    df_val = pd.concat([x_val, y_val], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    return df_train, df_test, df_val

def write_data_to_dir(df_train, df_test, df_val):
    if not os.path.exists('../splits/'):
        os.makedirs('../splits/')

    if not os.path.exists('../splits/baseline/'):
        os.makedirs('../splits/baseline/')

    if not os.path.exists('../splits/baseline/train/'):
        os.makedirs('../splits/baseline/train/')

    if not os.path.exists('../splits/baseline/validation/'):
        os.makedirs('../splits/baseline/validation/')
    if not os.path.exists('../splits/baseline/test/'):
        os.makedirs('../splits/baseline/test/')

    for i, row in df_train.iterrows():
        path = row['relative_path']
        shutil.copy(path, '../splits/baseline/train/')

    for i, row in df_test.iterrows():
        path = row['relative_path']
        shutil.copy(path, '../splits/baseline/test/')

    for i, row in df_val.iterrows():
        path = row['relative_path']
        shutil.copy(path, '../splits/baseline/validation/')

if __name__ == "__main__":
    df = pd.read_csv('../data/fish_lookup_table.csv')
    tr, te, val = create_train_test_val_splits(df)
    write_data_to_dir(tr, te, val)