from torch.optim import AdamW


class ResNetTrainingConfig:
    def __init__(self, lr=1e-3, weight_decay=1e-2, epochs=20, ft_epochs=10):
        self.lr = lr,
        self.weight_decay = weight_decay
        self.optimizer = AdamW(lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.ft_epochs = 10
