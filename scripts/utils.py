import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def plot_distribution(source_dir, dataset='imagewoof2-160'):
    data_dir = f'{source_dir}/{dataset}'
    filepath = f'{source_dir}/label_info.csv'
    label_info = pd.read_csv(filepath)
    for split in ['train', 'val']:
        labels = sorted(os.listdir(f'{data_dir}/{split}'))
        class2num = dict()
        for label in labels:
            label_dir = f'{data_dir}/{split}/{label}'
            name = label_info['name'][label_info['label'] == label].item()
            class2num[name] = len(os.listdir(label_dir))
        unique, counts = list(class2num.keys()), class2num.values()
        plt.barh(unique, counts, label=split)

    plt.title('Class Distribution')
    plt.xlabel('# images')
    plt.ylabel('Class')
    plt.legend()
    plt.show()


def show_rand(source_dir, dataset='imagewoof2-160'):
    np.random.seed(42)

    data_dir = f'{source_dir}/{dataset}'
    split = 'train'

    filepath = f'{source_dir}/label_info.csv'
    label_info = pd.read_csv(filepath)

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
    labels = sorted(os.listdir(f'{data_dir}/{split}'))
    fig.subplots_adjust(wspace=0)

    for i, ax in enumerate(axs.ravel()):
        label = labels[i]
        label_dir = f'{data_dir}/{split}/{label}'
        filenames = sorted(os.listdir(label_dir))

        image_id = np.random.randint(len(filenames))
        image_filepath = f'{label_dir}/{filenames[image_id]}'
        img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        name = label_info['name'][label_info['label'] == label].item()

        ax.imshow(img)
        ax.set_title(name)
        ax.set_axis_off()


def vis_aug(dataset):
    np.random.seed(42)

    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
    for ax in axs.ravel():
        idx = np.random.randint(len(dataset))
        image, _ = dataset[idx]
        ax.imshow(image)
        ax.set_axis_off()