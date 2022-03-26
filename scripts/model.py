import torchvision
import torch.nn as nn


def get_model(model_name, num_classes):
    model = torchvision.models.__dict__[model_name](pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
