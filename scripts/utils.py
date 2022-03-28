import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


def plot_conf_mtrx(gt, pred, target_names):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(gt, pred)
    cm = ConfusionMatrixDisplay(cm, display_labels=target_names)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.xticks(rotation=45)


def plot_class_dist(data_dir):
    for split in ['train', 'val']:
        labels = sorted(LABEL2ID_NAME.keys())
        class2num = dict()
        for label in labels:
            label_dir = f'{data_dir}/{split}/{label}'
            name = LABEL2ID_NAME[label][1]
            class2num[name] = len([x for x in os.listdir(label_dir) if '.JPEG' in x])
        unique, counts = list(class2num.keys()), class2num.values()
        plt.barh(unique, counts, label=split)

    plt.title('Distribution of classes')
    plt.xlabel('# images')
    plt.ylabel('class')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_size_dist(data_dir):
    height = dict()
    width = dict()
    labels = sorted(LABEL2ID_NAME.keys())
    for split in ['train', 'val']:
        height[split] = []
        width[split] = []
        for label in labels:
            label_dir = f'{data_dir}/{split}/{label}'
            filenames = sorted([x for x in os.listdir(label_dir) if '.JPEG' in x])
            for filename in filenames:
                image_filepath = f'{label_dir}/{filename}'
                img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
                h, w, c = img.shape
                height[split].append(h)
                width[split].append(w)

    min_h = min(min(height['train']), min(height['val']))
    max_h = max(max(height['train']), max(height['val']))
    print(f'height = [{min_h}, {max_h}]')
    min_w = min(min(width['train']), min(width['val']))
    max_w = max(max(width['train']), max(width['val']))
    print(f'width = [{min_w}, {max_w}]')

    fig, axs = plt.subplots(2, figsize=(16, 8))
    for split in ['train', 'val']:
        axs[0].hist(height[split], bins=50, label=split)
        axs[0].set_title('Distribution of height')
        axs[0].legend()
        axs[0].set_ylabel('# images')
        axs[0].set_xlabel('height')
        axs[1].hist(width[split], bins=50, label=split)
        axs[1].set_title('Distribution of width')
        axs[1].legend()
        axs[1].set_ylabel('# images')
        axs[1].set_xlabel('width')

    plt.tight_layout()
    plt.show()


def show_rand_img(data_dir, split='train'):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
    labels = sorted(LABEL2ID_NAME.keys())
    fig.subplots_adjust(wspace=0)
    for i, ax in enumerate(axs.ravel()):
        label = labels[i]
        label_dir = f'{data_dir}/{split}/{label}'
        filenames = sorted([x for x in os.listdir(label_dir) if '.JPEG' in x])

        image_id = np.random.randint(len(filenames))
        image_filepath = f'{label_dir}/{filenames[image_id]}'
        img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        name = LABEL2ID_NAME[label][1]

        ax.imshow(img)
        ax.set_title(name)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def show_aug_img(dataset):
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
    idx = np.random.randint(len(dataset))
    for ax in axs.ravel():
        image, _ = dataset[idx]
        ax.imshow(image)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
