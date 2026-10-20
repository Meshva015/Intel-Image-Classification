
import torch.nn as nn
from torchvision import models


def create_resnet18(num_classes=6, pretrained=True):
    """
    Create a ResNet18 model for Intel image classification.
    Supports feature extraction and fine-tuning.
    """

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final layer for our 6 image classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model):
    """Freeze all layers except the final classification layer."""

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_all_layers(model):
    """Allow all model layers to be trained during fine-tuning."""

    for param in model.parameters():
        param.requires_grad = True

    return model
