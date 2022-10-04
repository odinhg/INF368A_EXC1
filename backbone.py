import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights


class BackBone(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()
        self.number_of_classes = number_of_classes

        #Use EfficientNet V2 with small weights as base
        pretrained = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #Cut off last layers. Input (-1, 3, 300, 300), Output (-1, 1280, 10, 10) 
        self.feature_extractor = nn.Sequential(*list(pretrained.children())[:-2])
        for param in self.feature_extractor.parameters():
                param.requires_grad = False
        #Extra conv layers. Output (-1, 640, 5, 5)
        self.extra_layers = nn.Sequential(
                    nn.Conv2d(
                        in_channels=1280,
                        out_channels=640,
                        kernel_size=3,
                        stride=1,
                        padding=1
                        ),
                    #nn.BatchNorm2d(640),
                    #nn.ReLU(),
                    #nn.Conv2d(
                    #    in_channels=640,
                    #    out_channels=640,
                    #    kernel_size=3,
                    #    stride=1,
                    #    padding=1
                    #    ),
                    nn.BatchNorm2d(640),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=640,
                        out_channels=320,
                        kernel_size=3,
                        stride=1,
                        padding=1
                        ),
                    nn.BatchNorm2d(320),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
        #Fully connected layers. Input 8000, Output number of classes
        self.fc1 = nn.Sequential(
                nn.Linear(2880, 500), #8000 1000
                nn.ReLU(),
                nn.Dropout(0.2))
        self.fc2 = nn.Sequential(
                nn.Linear(500, self.number_of_classes))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.extra_layers(x)
        x = torch.flatten(x, start_dim = 1)
        a = self.fc1(x)
        x = self.fc2(a)
        return x, a
