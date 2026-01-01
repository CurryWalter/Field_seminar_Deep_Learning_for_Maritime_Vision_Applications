import numpy as np
import torch
import warnings
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)

def train_one_epoch(model, loss_fn, optimizer, dataloader, device):
    """
    Train a pytorch model for one epoch
    :param model:
    :param loss_fn:
    :param optimizer:
    :param dataloader:
    :param device:
    :return:
    """
    cum_loss = 0

    # set model mode to train
    model.train()

    for imgs, labels in tqdm(dataloader):
        # bring image to gpu if available
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        optimizer.zero_grad()

        # compute loss and do backward propagation
        loss = loss_fn(outputs, labels)
        loss.backward()

        # adjust model weights
        optimizer.step()

        cum_loss += loss.item()

    mean_loss = cum_loss / len(dataloader)
    return mean_loss



def validate(model, loss_fn, dataloader, device):
    """
    Validate a pytorch model for one epoch
    :param model: pytorch model
    :param dataloader: pytorch dataloader
    :param device: 'cpu' or 'cuda'
    :return: mean accuracy score over all batches
    """

    # set model mode to eval
    model.eval()

    cum_acc = 0
    cum_loss = 0
    for imgs, labels in dataloader:

        # put images, labels to gpu if available
        imgs = imgs.to(device)
        labels = labels.to(device)

        # disabling calculate gradients for inference
        with torch.no_grad():
            outputs = model(imgs)

        loss = loss_fn(outputs, labels)

        # convert outputs back to cpu
        outputs = outputs.detach().cpu().numpy()
        labels = labels.cpu().numpy()

        outputs = np.argmax(outputs, axis=1)


        # calculate accuracy
        accuracy = balanced_accuracy_score(labels, outputs)

        cum_loss += loss.item()
        cum_acc += accuracy

    mean_loss = cum_loss / len(dataloader)
    mean_acc = cum_acc / len(dataloader)

    return mean_acc, mean_loss
