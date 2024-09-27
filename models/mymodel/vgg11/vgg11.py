import torch
import torch.nn as nn
import torchvision.models as models

# from torchsummary import summary

import ttnn


class SimpleVGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleVGG, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolutional Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),  # Assuming input size is 224x224
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


class VGG11:
    def __init__(self, device, param, num_classes=1000):
        self.device = device

        breakpoint()

    def foward(self, x):
        return 0


if __name__ == "__main__":
    num_classes = 1000
    device_id = 4

    device = ttnn.open_device(device_id=device_id)

    vgg_torch = SimpleVGG(num_classes)
    parameters = vgg_torch.state_dict()
    vgg_ttnn = VGG11(device, parameters, num_classes)
