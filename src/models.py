import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights


def get_ResNet50(num_classes):
    """

    :param num_classes: int: Amount of classes
    :return:
    """

    # get pretrained model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # freeze base model layers for initial training
    for param in model.parameters():
        param.requires_grad = False

    # change classification head
    num_in_features_for_old_classification = model.fc.in_features
    new_head = nn.Linear(num_in_features_for_old_classification, num_classes)
    model.fc = new_head

    return model

def unfreeze_layers(model):
    for param in model.parameters():
        param.requires_grad = True