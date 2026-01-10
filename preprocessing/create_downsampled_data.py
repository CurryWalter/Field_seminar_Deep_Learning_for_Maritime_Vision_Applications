import pandas as pd
import os
import shutil

def create_downsampled_data():
    data = pd.read_csv('../splits/trajectory/train.csv')
    df = pd.DataFrame(data)

    df['label2'] = df['relative_path'].str.extract(r'fish_(\d+)_')[0]

    sampled_dfs = []
    unique_labels = df['label'].unique()
    for label in unique_labels:
        group = df[df['label'] == label]

        group_sampled = group.groupby('label2', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 100), random_state=0) if len(x) >= 100 else x
        , include_groups=False)
        # i have to double sample bcs .sample behaves weird idk should be fine xd
        if len(group_sampled) > 100:
            group_sampled = group_sampled.sample(n=100, random_state=0)


        sampled_dfs.append(group_sampled)

    return pd.concat(sampled_dfs)


def copy_and_rename_trajectory(source_dir='../splits/trajectory', dest_dir='../splits/downsampled'):

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)  # Source path
        d = os.path.join(dest_dir, item)  # Destination path
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=False, ignore=None)
        else:
            shutil.copy2(s, d)



def create_downsampled_dir(data):
    data.to_csv('../splits/downsampled/train.csv')
    shutil.rmtree('../splits/downsampled/train')
    os.makedirs('../splits/downsampled/train/')
    for i, row in data.iterrows():

        path = row['relative_path']
        shutil.copy(path, '../splits/downsampled/train/')
