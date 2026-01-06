import random

import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from src.dataset import FishyDataset



def apply_random_augmentation(image, seed, rand=.8):
    augmentations = []
    random.seed(seed)

    if random.random() < rand:
        angle = random.randint(-90, 90)
        augmentations.append(lambda img: img.rotate(angle))

    if random.random() < rand:
        augmentations.append(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))

    if random.random() < rand:
        brightness = 0.1 + (random.random() * 0.3)
        contrast = 0.1 + (random.random() * 0.3)
        # gurantees brightness and contrast to be between .1 and .4
        jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        augmentations.append(jitter)

    if random.random() < rand:
        max_shift = 15
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        augmentations.append(lambda img: img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty)))

    if random.random() < rand:

        max_shear = 0.2
        shear = random.uniform(-max_shear, max_shear)
        augmentations.append(lambda img: img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0)))

    for augment in augmentations:
        image = augment(image)

    return image

