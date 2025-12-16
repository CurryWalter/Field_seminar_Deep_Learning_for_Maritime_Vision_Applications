import torch
import os
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchvision.transforms import v2


class BasicTransforms:
    def __init__(self, image_size=(100, 100)):
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(image_size),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class ResNetTrainingConfig:
    def __init__(self, model_parameters, lr=1e-3, weight_decay=1e-2, loss_fn=CrossEntropyLoss(), batch_size=16, base_epochs=10, ft_epochs=10, image_size=(100, 100), transforms=BasicTransforms, run_name=None):
        self.model_parameters = model_parameters
        self.lr = lr,
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.optimizer = AdamW(params=model_parameters, lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.base_epochs = base_epochs
        self.ft_epochs = ft_epochs
        self.image_size = image_size
        self.transforms = transforms(self.image_size).transforms
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
        path = os.path.join(path_to_models_dir, f"ResNet50_{self.run_name}")
        return path

    def update_lr(self, new_lr):
        self.lr = new_lr
        self.optimizer = AdamW(params=self.model_parameters, lr=self.lr, weight_decay=self.weight_decay)
