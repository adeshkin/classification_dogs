import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
import torch.nn as nn


UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_img(filepath, img_size=(160, 160)):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image = transform(image=img)['image']

    return image.unsqueeze(0)


def get_model(model_name, num_classes):
    model = torchvision.models.__dict__[model_name](pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_name='resnet18',
               num_classes=10,
               model_chkpt='checkpoint/resnet18.pth'):

    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_chkpt, map_location=torch.device('cpu')))
    model.eval()

    return model





