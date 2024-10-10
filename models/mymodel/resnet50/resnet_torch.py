import torch
import torch.nn as nn
import torchvision.models as models

num_classes = 1000
RES_TORCH = models.resnet50(pretrained=True).eval()

if __name__ == "__main__":
    print(RES_TORCH)
    breakpoint()
