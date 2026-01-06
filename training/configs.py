import random

import torch
import os
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchvision.transforms import v2



def base_transforms(image_size=(100, 100)):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(image_size),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transforms


def aug_transforms(image_size=(100, 100), seed=0, rand=0.8):
    random.seed(seed)
    augmentations = []

    augmentations.append(v2.ToImage())
    augmentations.append(v2.ToDtype(torch.float32, scale=True))

    if random.random() < rand:
        angle = random.randint(-90, 90)
        augmentations.append(v2.RandomRotation(angle))

    if random.random() < rand:
        augmentations.append(v2.RandomHorizontalFlip())

    if random.random() < rand:
        brightness = 0.1 + (random.random() * 0.3)
        contrast = 0.1 + (random.random() * 0.3)
        jitter = v2.ColorJitter(brightness=brightness, contrast=contrast)
        augmentations.append(jitter)

    if random.random() < rand:
        max_shift = 15
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        aug_transform = v2.RandomAffine(0, translate=(tx / image_size[0], ty / image_size[1]))
        augmentations.append(aug_transform)

    if random.random() < rand:
        max_shear = 0.2
        shear = random.uniform(-max_shear, max_shear)
        augmentations.append(v2.RandomAffine(0, shear=shear))

    augmentations.append(v2.Resize(image_size))
    augmentations.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    return v2.Compose(augmentations)


class BaseTrainingConfig:
    def __init__(self, model_type, model_parameters, lr=1e-3, weight_decay=1e-2, loss_fn=CrossEntropyLoss(), batch_size=16, base_epochs=10, ft_epochs=10, image_size=(100, 100), transforms=base_transforms, run_name=None):
        self.model_type = model_type
        self.model_parameters = model_parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.optimizer = AdamW(params=model_parameters, lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.base_epochs = base_epochs
        self.ft_epochs = ft_epochs
        self.image_size = image_size
        self.transforms = transforms(self.image_size)
        self.run_name = run_name


    def get_train_params(self):
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'loss_fn': self.loss_fn,
            'base_epochs': self.base_epochs,
            'ft_epochs': self.ft_epochs,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer,
            'image_size': self.image_size,
            'transforms': self.transforms
        }

    def get_save_path(self, path_to_models_dir):
        if self.model_type.lower() == 'resnet':
            path = os.path.join(path_to_models_dir, f"ResNet50_{self.run_name}")
        else:
            path = os.path.join(path_to_models_dir, f"ViTb16_{self.run_name}")
        return path

    def update_lr(self, new_lr, model_parameters):
        self.lr = new_lr
        self.model_parameters = model_parameters
        self.optimizer = AdamW(params=self.model_parameters, lr=self.lr, weight_decay=self.weight_decay)
