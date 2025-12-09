import torch


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

    for imgs, labels in dataloader:
        # bring image to gpu if available
        imgs.to(device)
        labels.to(device)

        outputs = model(imgs)
        print(outputs)

        optimizer.zero_grad()

        # compute loss and do backward propagation
        loss = loss_fn(outputs, labels)
        loss.backward()

        print(loss.item())
        # adjust model weights
        optimizer.step()

        cum_loss += loss.item()

    mean_loss = cum_loss / len(dataloader)
    return mean_loss



def validate(model, dataloader, device):
    """
    Validate a pytorch model for one epoch
    :param model: pytorch model
    :param dataloader: pytorch dataloader
    :param device: 'cpu' or 'cuda'
    :return: mean accuracy score over all batches
    """

    # set model mode to eval
    model.eval()

    for imgs, labels in dataloader:
        # images to gpu if available
        imgs.to(device)

        # disabling calculating gradients for inference
        with torch.no_grad():
            outputs = model(imgs)

        # convert outputs back to cpu
        outputs = outputs.detach().cpu().numpy()
