import torchvision
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, num_classes, in_ch=3, out_chs=[8, 16, 32, 64]):
        super(MyNet, self).__init__()
        self.layers = []
        for i, out_ch in enumerate(out_chs):
            if i == 0:
                in_ch = in_ch
            else:
                in_ch = out_chs[i-1]

            self.layers[i] = nn.Sequential(
                nn.Conv2d(in_ch, out_ch[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.fc = nn.Linear(7 * 7, num_classes)

    def forward(self, x):
        for i in range(len(self.layers)):
            out = self.layers[i](x)

        print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model(model_name, num_classes):
    if model_name in ['resnet18', 'efficientnet_b0']:
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
