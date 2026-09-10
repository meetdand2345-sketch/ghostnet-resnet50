import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def build_model(pretrained=True):
    weights=DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model=deeplabv3_resnet50(weights=weights)
    model.classifier[4]=nn.Conv2d(256,1,1)
    if model.aux_classifier is not None:
        model.aux_classifier[4]=nn.Conv2d(256,1,1)
    return model
