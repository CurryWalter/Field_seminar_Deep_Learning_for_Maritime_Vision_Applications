import os
import pandas as pd
from torch.utils.data import Dataset
from src.helper_functions import match_name_to_label
from PIL import Image


class FishyDataset(Dataset):
    """
    Custom Pytorch Dataset class for load images from Fish4Knowledge Dataset
    """
    def __init__(self, img_dir, annotation_file, transforms):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        label = int(match_name_to_label(img_name)[-2:]) - 1
        if self.transforms:
            img = self.transforms(img)

        return img, label
