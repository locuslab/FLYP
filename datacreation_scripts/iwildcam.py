import os

import pandas as pd
import argparse


def main(args):
    iwildcam_template = [
        lambda c: f"a photo of {c}.", lambda c: f"{c} in the wild."
    ]
    df = pd.read_csv(args.metadata)
    df = df[(df['split'] == 'train') & (df['y'] < 99999)][['filename', 'y']]

    label_to_name = pd.read_csv(args.english_label_path)
    label_to_name = label_to_name[label_to_name['y'] < 99999]

    assert len(df) == 129809, 'number of samples incorrect'

    label_to_name['prompt1'] = label_to_name['english'].map(
        iwildcam_template[0])
    label_to_name['prompt2'] = label_to_name['english'].map(
        iwildcam_template[1])

    df1 = pd.merge(df, label_to_name[['y', 'prompt1']],
                   on='y').rename({'prompt1': 'title'}, axis='columns')
    df2 = pd.merge(df, label_to_name[['y', 'prompt2']],
                   on='y').rename({'prompt2': 'title'}, axis='columns')

    assert len(df1) == 129809, 'number of samples incorrect'
    assert len(df2) == 129809, 'number of samples incorrect'

    df_final = pd.concat((df1, df2))[['filename', 'title', 'y']]

    del df1
    del df2
    del df

    df_final['filename'] = df_final['filename'].map(
        lambda x: f'{args.data_dir}/{x}')
    df_final = df_final.rename({
        'filename': 'filepath',
        'y': 'label'
    },
                               axis='columns')[['title', 'filepath', 'label']]

    assert len(df_final) == 129809 * 2, 'number of samples incorrect'

    df_final.to_csv(args.save_file, sep='\t', index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--save_file',
                        default='./datasets/csv/iwildcam_v2.0/train.csv')
    parser.add_argument('--english_label_path',
                        default='./src/datasets/iwildcam_metadata/labels.csv')
    parser.add_argument('--metadata',
                        default='./datasets/data/iwildcam_v2.0/metadata.csv')
    parser.add_argument('--data_dir',
                        default='./datasets/data/iwildcam_v2.0/train')
    args = parser.parse_args()

    os.makedirs('./datasets/csv/iwildcam_v2.0', exist_ok=True)

    main(args)
