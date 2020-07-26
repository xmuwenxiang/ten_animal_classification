#! -*- coding: utf-8 -*-
from torchvision.models import vgg16_bn
import torch.nn as nn
import torch


class Vgg16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Vgg16, self).__init__()
        self.features = vgg16_bn(pretrained=pretrained).features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
