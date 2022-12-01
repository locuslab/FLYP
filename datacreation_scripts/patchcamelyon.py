import argparse
import os
import shutil

import h5py
from PIL import Image
from tqdm import tqdm

classes = ['lymph node', 'lymph node containing metastatic tumor tissue']

templates = [lambda c: f'this is a photo of {c}']


def main(args):

    for typ in ['train', 'valid', 'test']:
        name = typ
        if typ == 'valid':
            name = 'val'
        with h5py.File(
                os.path.join(args.data_dir,
                             f'camelyonpatch_level_2_split_{typ}_x.h5'),
                "r") as f:
            a_group_key = list(f.keys())[0]
            x = f[a_group_key][()]

        with h5py.File(
                os.path.join(args.data_dir,
                             f'camelyonpatch_level_2_split_{typ}_y.h5'),
                "r") as f:
            a_group_key = list(f.keys())[0]
            y = f[a_group_key][()].reshape(-1)

        assert x.shape[0] == y.shape[0]

        shutil.rmtree(os.path.join(args.data_dir, name),
                      ignore_errors=True,
                      onerror=None)

        os.makedirs(os.path.join(args.data_dir, name), exist_ok=True)
        os.makedirs(os.path.join(args.data_dir, name, '0'), exist_ok=True)
        os.makedirs(os.path.join(args.data_dir, name, '1'), exist_ok=True)

        for i in tqdm(range(x.shape[0])):
            im = x[i, :, :, :]
            label = int(y[i])
            filename = os.path.join(args.data_dir, name, f'{label}',
                                    f'{i}.jpeg')

            im = Image.fromarray(im)
            im.save(filename)

    classes_in_dir = sorted(
        next(os.walk(os.path.join(args.data_dir, 'train')))[1])
    print(classes_in_dir)
    assert len(classes) == len(classes_in_dir), 'num class mismatch'

    os.makedirs(os.path.join(args.save_dir, args.data_name), exist_ok=True)
    with open(os.path.join(args.save_dir, args.data_name, 'train.csv'),
              'w') as f:
        f.write('title\tfilepath\n')
        for i, dir_name in enumerate(classes_in_dir):
            directory = os.path.join(args.data_dir, 'train', dir_name)
            for file in os.listdir(directory):
                assert 'jpg' in file or 'jpeg' in file, f'extension mismatch {file} {directory}'
                full_path = os.path.join(args.data_dir, 'train', dir_name,
                                         file)
                for template in templates:
                    f.write(f'{template(classes[i])}\t{full_path}\n')

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--save_dir', default='./datasets/csv')
    parser.add_argument('--data_dir', default='./datasets/data')
    parser.add_argument('--data_name', default='patchcamelyon')

    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.data_name)

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
