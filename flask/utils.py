import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
import torch.nn as nn


UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


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





