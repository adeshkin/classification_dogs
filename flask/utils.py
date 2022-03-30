import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn


LABEL2ID_NAME = {
    'n02086240': (0, 'Shih-Tzu'),
    'n02087394': (1, 'Rhodesian_ridgeback'),
    'n02088364': (2, 'beagle'),
    'n02089973': (3, 'English_foxhound'),
    'n02093754': (4, 'Border_terrier'),
    'n02096294': (5, 'Australian_terrier'),
    'n02099601': (6, 'golden_retriever'),
    'n02105641': (7, 'Old_English_sheepdog'),
    'n02111889': (8, 'Samoyed'),
    'n02115641': (9, 'dingo')
}

ID2NAME = {idx: name for idx, name in LABEL2ID_NAME.values()}


def prepare_img(img, img_size=(160, 160)):
    transform = T.Compose([T.ToTensor(),
                           T.Resize(img_size),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    image = transform(img)

    return image.unsqueeze(0)


def get_model(model_name, num_classes):
    model = torchvision.models.__dict__[model_name](pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_name='resnet18',
               num_classes=10,
               model_chkpt='resnet18_best.pth'):
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_chkpt, map_location=torch.device('cpu')))
    model.eval()

    return model
