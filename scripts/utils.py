import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def stats(data_dir):
    for mode in ['train', 'val']:
        labels = sorted(os.listdir(f'{data_dir}/{mode}'))
        label2num = dict()
        for label in labels:
            label_dir = f'{data_dir}/{mode}/{label}'
            label2num[label] = len(os.listdir(label_dir))

        unique, counts = list(label2num.keys()), label2num.values()
        plt.barh(unique, counts)

    plt.title('Class Frequency')
    plt.xlabel('Frequency')
    plt.ylabel('Class')

    plt.show()


def show_random(data_dir):
    num_labels = 10
    num_train_imgs, num_val_imgs = 3, 2
    num_images = num_train_imgs + num_val_imgs
    fig, axs = plt.subplots(num_labels, num_images, figsize=(20, 10))
    fig.subplots_adjust(wspace=0)
    labels = sorted(os.listdir(f'{data_dir}/train'))
    for i, ax in enumerate(axs.ravel()):
        mode_id = i % num_images
        if mode_id < num_train_imgs:
            mode = 'train'
        else:
            mode = 'val'
        label_id = i // num_images
        label = labels[label_id]

        label_dir = f'{data_dir}/{mode}/{label}'
        filenames = sorted(os.listdir(label_dir))

        image_id = np.random.randint(len(filenames))

        image_filepath = f'{label_dir}/{filenames[image_id]}'
        img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        # ax.set_title(f'{mode} {label}')
        # ax.set_axis_off()
    plt.show()