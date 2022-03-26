import torch
import torchvision
import torch.nn as nn


def get_model(model_name, num_classes):
    model = torchvision.models.__dict__[model_name](pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def load_model(model_name='resnet18', num_classes=10,
               chkpt_dir='/Users/adyoshkin/Desktop/classification_dogs/checkpoints'):
    model_chkpt = f'{chkpt_dir}/{model_name}.pth'
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_chkpt, map_location=torch.device('cpu')))
    model.eval()

    return model
