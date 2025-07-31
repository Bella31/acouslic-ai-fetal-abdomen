import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_finetune_resnet_model(num_classes=3):
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=(7, 7),
                            stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
