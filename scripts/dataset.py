import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class DogDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.filenames = df['filename'].tolist()
        self.labels = df['label'].tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = f'{self.img_dir}/{self.filenames[idx]}'
        label = self.labels[idx]

        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=img)["image"]

        return image, label


def get_transforms(img_size):
    transforms = dict()
    train_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes=2, min_height=10, max_height=20, min_width=10, max_width=20, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transforms['train'] = train_transform
    transforms['val'] = val_transform

    return transforms


def get_dl(data_dir, splits, img_size, batch_size):
    dl = dict()
    transforms = get_transforms(img_size)
    if 'dev' in splits:
        assert 'train1' in splits, 'Training on train1, validation on dev, test on val'

    for split in splits:
        if 'train' in split:
            transform = transforms['train']
        if split == 'dev':
            transform = transforms['val']

        df = pd.read_csv(f'{data_dir}/{split}.csv')
        img_dir = f'{data_dir}/{split}'
        ds = DogDataset(df, img_dir, transform)
        dl[split] = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dl


def get_ds(data_dir, img_size=(160, 160)):
    ds = dict()
    transforms = get_transforms(img_size)
    for split in ['train', 'val']:
        df = pd.read_csv(f'{data_dir}/{split}.csv')
        img_dir = f'{data_dir}/{split}'
        ds[split] = DogDataset(df, img_dir, transforms[split])

    return ds



