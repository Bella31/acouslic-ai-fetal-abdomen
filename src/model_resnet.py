import torch
import torch.nn as nn
import torchvision.models as models

def get_finetune_resnet_model(num_classes=3, pretrained=True, grayscale=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=pretrained)

    if grayscale:
        # Modify first conv layer to accept 1 channel instead of 3
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3), bias=False)

    # Replace fully connected layer for classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)

def freeze_layers(model, freeze_until="layer4"):
    freeze = True
    for name, param in model.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until in name:
            freeze = False
