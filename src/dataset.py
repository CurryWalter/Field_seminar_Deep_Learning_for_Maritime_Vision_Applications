import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class FishyDataset(Dataset):
    """
    Custom Pytorch Dataset class for load images from Fish4Knowledge Dataset
    """
    def __init__(self, annotation_file, transforms, img_dir=None):
        """
        :param annotation_file: Path to the annotation file
        :param transforms: Pytorch transforms
        :param img_dir: Path to the images directory
        """
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations['image_name'].tolist())

    def __getitem__(self, idx):
        if self.img_dir is None:
            img_path = self.annotations['relative_path'][idx]
        else:
            img_name = self.annotations['image_name'][idx]
            img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        label = int(self.annotations['label'][idx][-2:]) - 1

        if self.transforms:
            img = self.transforms(img)

        return img, label
