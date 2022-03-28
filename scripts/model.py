import torchvision
import torch.nn as nn


def get_model(model_name, num_classes):
    if model_name in torchvision.models.__dict__:
        model = torchvision.models.__dict__[model_name](pretrained=True)
        if 'resnet' in model_name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'efficientnet' in model_name:
            model.fc = nn.Linear(1280, num_classes)
    else:
        raise NotImplementedError('Model is not implemented')

    return model
