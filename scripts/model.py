import torchvision
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(18 * 18 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model(model_name, num_classes):
    if model_name in torchvision.models.__dict__:
        model = torchvision.models.__dict__[model_name](pretrained=True)
        if 'resnet' in model_name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'efficientnet' in model_name:
            model.fc = nn.Linear(1280, num_classes)
    elif model_name == 'my_net':
        model = MyNet(num_classes)
    else:
        raise NotImplementedError('Model is not implemented')

    return model
