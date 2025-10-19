import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, pretrained=True):
        super(ResNetClassifier, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=pretrained)
        # Replace the final fully connected layer to match the number of classes
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through the ResNet model
        x = self.resnet(x)
        return x
