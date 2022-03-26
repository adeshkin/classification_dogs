import os
import pandas as pd


def main():
    data_dir = '/content/drive/MyDrive/classification_dogs/data/imagewoof2-160'
    splits = ['train', 'val']

    labels = sorted([x for x in os.listdir(f'{data_dir}/train')
                     if os.path.isdir(f'{data_dir}/train/{x}')])
    label2id = {label: i for i, label in enumerate(labels)}
    label2id_df = pd.DataFrame({'label': label2id.keys(),
                                'id': label2id.values()})
    label2id_df.to_csv(f'{data_dir}/label2id.csv', index=False, header=True)

    for split in splits:
        df = pd.DataFrame(columns=['filepath', 'label'])
        for label in labels:
            label_dir = f'{data_dir}/{split}/{label}'
            filenames = sorted([f'{label_dir}/{x}' for x in os.listdir(label_dir) if '.JPEG' in x])
            label_df = pd.DataFrame(filenames, columns=['filepath'])
            label_df['label'] = label2id[label]
            df = pd.concat([df, label_df], ignore_index=True)

        df.to_csv(f'{data_dir}/{split}.csv', index=False, header=True)


if __name__ == '__main__':
    main()
