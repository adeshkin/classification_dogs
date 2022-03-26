import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from utils import get_transform


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


def get_dls(data_dir, splits, batch_size):
    dls = dict()
    transform = get_transform()
    for split in splits:
        df = pd.read_csv(f'{data_dir}/{split}.csv')
        ds = DogDataset(df, transform[split])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=batch_size)
        dls[split] = dl

    return dls
