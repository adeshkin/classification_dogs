import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class DogDataset(Dataset):
    def __init__(self, df, transform=None):
        self.filepaths = df['filepath'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=img)["image"]

        return image, label


def get_transform():
    transform = dict()
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.CenterCrop(height=128, width=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transform['train'] = train_transform
    transform['val'] = val_transform

    return transform


def get_dls(data_dir, splits, batch_size):
    dls = dict()
    transform = get_transform()
    for split in splits:
        df = pd.read_csv(f'{data_dir}/{split}.csv')
        ds = DogDataset(df, transform[split])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=batch_size)
        dls[split] = dl

    return dls

