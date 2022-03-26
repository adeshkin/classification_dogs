import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from collections import defaultdict


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.metrics = None
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get_metrics(self):
        return {metric_name: metric["avg"] for (metric_name, metric) in self.metrics.items()}

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def plot_distribution(source_dir, dataset_name='imagewoof2-160'):
    data_dir = f'{source_dir}/{dataset_name}'
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


def show_rand_img(source_dir, dataset_name='imagewoof2-160', seed=42):
    np.random.seed(seed)

    data_dir = f'{source_dir}/{dataset_name}'
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


def vis_aug(dataset, seed=42):
    np.random.seed(seed)

    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
    for ax in axs.ravel():
        idx = np.random.randint(len(dataset))
        image, _ = dataset[idx]
        ax.imshow(image)
        ax.set_axis_off()
