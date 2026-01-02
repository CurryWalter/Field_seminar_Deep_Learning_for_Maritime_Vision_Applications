import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights


def prepare_transfer_learning_models(model, num_classes):
    """
    Prepare Pytorch model for transfer learning
    :param model: Pytorch model
    :param num_classes: int: Amount of classes
    :return: Pytorch model
    """
    # freeze base model layers for initial training
    for param in model.parameters():
        param.requires_grad = False

    # change classification head
    if isinstance(model, models.ResNet):
        num_in_features_for_old_classification = model.fc.in_features
        new_head = nn.Linear(num_in_features_for_old_classification, num_classes)
        model.fc = new_head
    else:
        num_in_features_for_old_classification = model.hidden_dim
        new_head = nn.Linear(num_in_features_for_old_classification, num_classes)
        model.heads.pop(0)
        model.heads.append(new_head)

    return model

def get_ResNet50(num_classes):
    """
    Returns a ResNet50 model
    :param num_classes: int: Amount of classes
    :return: ResNet50 model
    """

    # get pretrained model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    model = prepare_transfer_learning_models(model, num_classes)

    return model


def get_ViTb16(num_classes, image_size):
    """
    Returns a ViTb16 model
    :param num_classes: int: Amount of classes
    :return: ViTb16 model
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1, image_size=image_size)
    model = prepare_transfer_learning_models(model, num_classes)

    return model

def unfreeze_layers(model):
    if isinstance(model, models.ResNet):
        for param in model.parameters():
            param.requires_grad = True
    else:
        for layer_block in model.encoder.layers[-3:]:
            for param in layer_block.parameters():
                param.requires_grad = True
