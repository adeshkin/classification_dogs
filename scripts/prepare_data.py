import os
import pandas as pd


def main(source_dir, dataset_name='imagewoof2-160', map_filename='map_clsloc.txt'):
    data_dir = f'{source_dir}/{dataset_name}'
    map_filepath = f'{source_dir}/{map_filename}'
    splits = ['train', 'val']

    labels = sorted([x for x in os.listdir(f'{data_dir}/train')
                     if os.path.isdir(f'{data_dir}/train/{x}')])

    with open(map_filepath) as f:
        all_label2name = {line.split()[0]: line.split()[-1] for line in f.readlines()}

    label2id = {label: i for i, label in enumerate(labels)}
    label2name = {label: all_label2name[label] for label in labels}

    label2id_df = pd.DataFrame({'label': label2id.keys(),
                                'id': label2id.values(),
                                'name': label2name.values()})
    label2id_df.to_csv(f'{source_dir}/label_info.csv', index=False, header=True)

    for split in splits:
        df = pd.DataFrame(columns=['filename', 'label'])
        for label in labels:
            label_dir = f'{data_dir}/{split}/{label}'
            filenames = sorted([x for x in os.listdir(label_dir) if '.JPEG' in x])
            label_df = pd.DataFrame(filenames, columns=['filename'])
            label_df['label'] = label2id[label]
            df = pd.concat([df, label_df], ignore_index=True)

        df.to_csv(f'{source_dir}/{split}.csv', index=False, header=True)


if __name__ == '__main__':
    main(source_dir='../data')
