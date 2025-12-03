import os
import shutil

import pandas as pd
from create_lookup_table import create_lookup_table
from sklearn.model_selection import train_test_split

def create_train_test_val_splits(df=create_lookup_table(), train_ratio=4/7, validation_ratio=1/7, test_ratio=2/7):
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, 'relative_path'], df.loc[:, 'lable'], test_size=1 - train_ratio, stratify=df.loc[:, 'lable'], random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0)

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
        path = row['absolute_path']
        shutil.copy(path, '../splits/baseline/train/')

    for i, row in df_test.iterrows():
        path = row['absolute_path']
        shutil.copy(path, '../splits/baseline/test/')

    for i, row in df_val.iterrows():
        path = row['absolute_path']
        shutil.copy(path, '../splits/baseline/validation/')

if __name__ == "__main__":
    tr, te, val = create_train_test_val_splits()
    write_data_to_dir(tr, te, val)