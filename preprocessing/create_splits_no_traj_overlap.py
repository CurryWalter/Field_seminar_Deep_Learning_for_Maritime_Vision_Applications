import os
import random
import shutil

import pandas as pd



def create_train_test_val_splits_traj_overlap(
        df,
        train_ratio=4/7,
        validation_ratio=1/7,
        test_ratio=2/7):
    # prioritises 4:2:1 and then that each trajectory doesnt overlap
    train, test, val = [], [], []
    for label in df['label'].unique():
        df_label = df[df['label'] == label]

        train_temp, val_temp, test_temp = [], [], []
        train_size = val_size = test_size = 0

        dataframe_length = len(df_label)
        train_target_size = train_ratio * dataframe_length
        val_target_size = validation_ratio * dataframe_length
        test_target_size = test_ratio * dataframe_length

        trajectories = list(df_label['trajectory'].unique())
        random.seed(0)
        random.shuffle(trajectories)

        for traj in trajectories:
            trajectory_series = df_label[df_label['trajectory'] == traj]
            trajectory_length = len(trajectory_series)

            train_difference = train_target_size - train_size
            val_difference   = val_target_size - val_size
            test_difference  = test_target_size - test_size

            if trajectory_length > max(train_difference, val_difference, test_difference):
                split_train = int(round(train_ratio * trajectory_length))
                split_test  = int(round(test_ratio * trajectory_length))
                split_val   = trajectory_length - split_train - split_test

                if split_train > 0:
                    train_temp.append(trajectory_series.iloc[:split_train])
                    train_size += split_train
                if split_test > 0:
                    test_temp.append(trajectory_series.iloc[split_train:split_train+split_test])
                    test_size += split_test
                if split_val > 0:
                    val_temp.append(trajectory_series.iloc[split_train+split_test:])
                    val_size += split_val
            else:
                if train_difference >= val_difference and train_difference >= test_difference:
                    train_temp.append(trajectory_series)
                    train_size += trajectory_length
                elif val_difference >= train_difference and val_difference >= test_difference:
                    val_temp.append(trajectory_series)
                    val_size += trajectory_length
                else:
                    test_temp.append(trajectory_series)
                    test_size += trajectory_length

        train.append(pd.concat(train_temp))
        test.append(pd.concat(test_temp))
        val.append(pd.concat(val_temp))

    return pd.concat(train), pd.concat(test), pd.concat(val)

def write_data_to_dir_traj_overlap(df_train, df_test, df_val):
    if not os.path.exists('../splits/'):
        os.makedirs('../splits/')

    if not os.path.exists('../splits/trajectory/'):
        os.makedirs('../splits/trajectory/')

    if not os.path.exists('../splits/trajectory/train/'):
        os.makedirs('../splits/trajectory/train/')

    if not os.path.exists('../splits/trajectory/validation/'):
        os.makedirs('../splits/trajectory/validation/')
    if not os.path.exists('../splits/trajectory/test/'):
        os.makedirs('../splits/trajectory/test/')

    for i, row in df_train.iterrows():
        path = row['relative_path']
        shutil.copy(path, '../splits/trajectory/train/')

    for i, row in df_test.iterrows():
        path = row['relative_path']
        shutil.copy(path, '../splits/trajectory/test/')

    for i, row in df_val.iterrows():
        path = row['relative_path']
        shutil.copy(path, '../splits/trajectory/validation/')

if __name__ == "__main__":
    df = pd.read_csv('../data/fish_lookup_table.csv')
    tr, te, val = create_train_test_val_splits_traj_overlap()
    write_data_to_dir_traj_overlap(tr, te, val)