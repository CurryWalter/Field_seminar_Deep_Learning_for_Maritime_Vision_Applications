import os
import shutil
from augment_training import *
import pandas as pd


def copy_and_rename_downsampled(source_dir='../splits/downsampled', dest_dir='../splits/downsampled_with_aug'):

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

def apply_aug():
    df = pd.read_csv('../splits/downsampled_with_aug/train.csv')
    for cls, count in df['label'].value_counts().items():
        while count < 100:
            difference = 100 - count
            augment_class(cls, difference, 'downsampled_with_aug')
            # this line below is necessary bcs .sample is weird imo
            count = pd.read_csv('../splits/downsampled_with_aug/train.csv')['label'].value_counts().loc[cls]
