import torch
import mlflow
from torch.utils.data import DataLoader
from torchvision import models
from src.engine import train_one_epoch, validate
from src.dataset import FishyDataset
from src.models import get_ResNet50, get_ViTb16, unfreeze_layers
from configs import BaseTrainingConfig

def main(model, image_size):

    model_type = 'resnet' if isinstance(model, models.ResNet) else 'vit'

    # get train config
    train_config = BaseTrainingConfig(model_type, model.parameters(), run_name='first_run_baseline_dataset', image_size=image_size)

    # use gpu if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # load dataset
    img_dir = "../data/fish_image/"
    train_file = "../splits/baseline/train.csv"
    val_file = "../splits/baseline/val.csv"

    # path for model save
    if isinstance(model, models.ResNet):
        model_path = "../models/ResNet50/"
    else:
        model_path = "../models/ViTb16/"


    train_dataset = FishyDataset(train_file, train_config.transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)

    val_dataset = FishyDataset(val_file, train_config.transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

    with mlflow.start_run(run_name=train_config.run_name):
        # log training parameters
        mlflow.log_params(train_config.get_train_params())

        model_info = mlflow.pytorch.log_model(model)

        # put model on device
        model.to(device)

        # for validation
        best_accuracy = 0

        for i in range(train_config.base_epochs + train_config.ft_epochs):
            print(f"Epoch {i+1}/{train_config.base_epochs + train_config.ft_epochs}")

            # unfreeze top layers during fine tuning
            if i == train_config.base_epochs:
                unfreeze_layers(model)
                train_config.update_lr(train_config.lr / 10, model.parameters())


            loss = train_one_epoch(model, train_config.loss_fn, train_config.optimizer, train_dataloader, device)
            mlflow.log_metric('train_loss', loss, step=i)

            print(f"Validating")

            accuracy, val_loss = validate(model, train_config.loss_fn, val_dataloader, device)
            print(accuracy, val_loss)
            mlflow.log_metrics({'val_accuracy': accuracy, 'val_loss': val_loss}, step=i)

            if best_accuracy < accuracy:
                torch.save(model.state_dict(), train_config.get_save_path(model_path))




if __name__ == "__main__":
    image_size = (224, 224)
    num_classes = 23

    # for Resnet
    # model = get_ResNet50(num_classes)

    # for ViT
    model = get_ViTb16(num_classes, image_size[0])

    main(model, image_size)