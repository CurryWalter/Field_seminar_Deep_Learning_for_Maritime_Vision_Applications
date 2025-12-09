import torch
import mlflow
from torch.utils.data import DataLoader
from src.engine import train_one_epoch, validate
from src.dataset import FishyDataset
from src.models import get_ResNet50
from configs import ResNetTrainingConfig

def main():
    # get model
    model = get_ResNet50(23)

    # get train config
    train_config = ResNetTrainingConfig(model.parameters(), run_name='first_run_baseline_dataset')

    # use gpu if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # load dataset
    img_dir_train = '../splits/baseline/train'
    img_dir_val = '../splits/baseline/validation'
    annotation_file = '../data/fish_lookup_table.csv'

    train_dataset = FishyDataset(img_dir_train, annotation_file, train_config.transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)

    val_dataset = FishyDataset(img_dir_val, annotation_file, train_config.transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

    with mlflow.start_run(run_name=train_config.run_name):
        # log training parameters
        mlflow.log_params(train_config.get_train_params())

        model_info = mlflow.pytorch.log_model(model)

        # put model on device
        model.to(device)

        for i in range(train_config.base_epochs):
            print(f"Epoch {i+1}/{train_config.base_epochs}")

            loss = train_one_epoch(model, train_config.loss_fn, train_config.optimizer, train_dataloader, device)
            mlflow.log_metric('train_loss', loss, step=i)

            accuracy, val_loss = validate(model, val_dataloader, device)
            mlflow.log_metrics({'val_accuracy': accuracy, 'val_loss': val_loss}, step=i)






if __name__ == "__main__":
    main()