from turtle import forward
import torch.nn as nn
import torchvision


class Resnet():
    def __init__(self, numOfClass):
        self.model = torchvision.models.resnet18(pretrained=True)

        num_ftrs = self.model.fc.in_features
        self.fc = nn.Linear(num_ftrs, numOfClass)
    
    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out