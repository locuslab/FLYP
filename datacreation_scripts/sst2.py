import argparse
import os

classes = [
    'negative',
    'positive'
]

templates = [
    lambda c: f'a {c} review of a movie.'
]


def main(args):

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
                assert 'jpg' in file or 'jpeg' in file or 'png' in file, f'extension mismatch {file} {directory}'
                full_path = os.path.join(args.data_dir, 'train', dir_name,
                                         file)
                for template in templates:
                    f.write(f'{template(classes[i])}\t{full_path}\n')

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--save_dir', default='./datasets/csv')
    parser.add_argument('--data_dir', default='./datasets/data')
    parser.add_argument('--data_name', default='sst2')

    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.data_name)

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
