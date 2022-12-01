import os
import argparse
from scipy.io import loadmat

import shutil

final_classes = [
    'air plant', 'alpine sea holly', 'anthurium', 'artichoke', 'azalea',
    'balloon flower', 'barbeton daisy', 'bearded iris', 'bee balm',
    'bird of paradise', 'bishop of llandaff', 'black-eyed susan',
    'blackberry lily', 'blanket flower', 'bolero deep blue', 'bougainvillea',
    'bromelia', 'buttercup', 'californian poppy', 'camellia', 'canna lily',
    'canterbury bells', 'cape flower', 'carnation', 'cautleya spicata',
    'clematis', "colt's foot", 'columbine', 'common dandelion', 'corn poppy',
    'cyclamen', 'daffodil', 'desert-rose', 'english marigold', 'fire lily',
    'foxglove', 'frangipani', 'fritillary', 'garden phlox', 'gaura', 'gazania',
    'geranium', 'giant white arum lily', 'globe flower', 'globe thistle',
    'grape hyacinth', 'great masterwort', 'hard-leaved pocket orchid',
    'hibiscus', 'hippeastrum', 'japanese anemone', 'king protea',
    'lenten rose', 'lotus', 'love in the mist', 'magnolia', 'mallow',
    'marigold', 'mexican aster', 'mexican petunia', 'monkshood', 'moon orchid',
    'morning glory', 'orange dahlia', 'osteospermum', 'oxeye daisy',
    'passion flower', 'pelargonium', 'peruvian lily', 'petunia',
    'pincushion flower', 'pink and yellow dahlia', 'pink primrose',
    'poinsettia', 'primula', 'prince of wales feathers', 'purple coneflower',
    'red ginger', 'rose', 'ruby-lipped cattleya', 'siam tulip', 'silverbush',
    'snapdragon', 'spear thistle', 'spring crocus', 'stemless gentian',
    'sunflower', 'sweet pea', 'sweet william', 'sword lily', 'thorn apple',
    'tiger lily', 'toad lily', 'tree mallow', 'tree poppy', 'trumpet creeper',
    'wallflower', 'water lily', 'watercress', 'wild pansy', 'windflower',
    'yellow iris'
]

open_ai_classes = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells',
    'sweet pea', 'english marigold', 'tiger lily', 'moon orchid',
    'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
    "colt's foot", 'king protea', 'spear thistle', 'yellow iris',
    'globe flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
    'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
    'stemless gentian', 'artichoke', 'sweet william', 'carnation',
    'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
    'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
    'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
    'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
    'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
    'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
    'pink and yellow dahlia', 'cautleya spicata', 'japanese anemone',
    'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum',
    'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania',
    'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
    'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
    'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow',
    'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum',
    'bee balm', 'air plant', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
    'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
    'blackberry lily'
]

templates = [
    lambda c: f'a photo of a {c}, a type of flower.'
]


def main(args):

    classes = [a.replace(' ', '_') for a in open_ai_classes]

    all_y = list(
        loadmat(os.path.join(args.data_dir, 'imagelabels.mat'))['labels'][0])

    all_splits = loadmat(os.path.join(args.data_dir, 'setid.mat'))

    train_id = list(all_splits['trnid'][0])
    val_id = list(all_splits['valid'][0])
    test_id = list(all_splits['tstid'][0])

    assert min(all_y) == 1, 'min value is not 1'

    for typ in ['train', 'val', 'test']:
        shutil.rmtree(os.path.join(args.data_dir, typ),
                      ignore_errors=True,
                      onerror=None)
        for name in classes:
            os.makedirs(os.path.join(args.data_dir, typ, name), exist_ok=True)
            if typ == 'train':
                X = train_id
            elif typ == 'val':
                X = val_id
            elif typ == 'test':
                X = test_id

        for i, number in enumerate(X):
            value = str(number).zfill(5)
            fname = f'image_{value}.jpg'
            os.symlink(
                os.path.join(args.data_dir, 'all_images', fname),
                os.path.join(args.data_dir, typ,
                             classes[all_y[number - 1] - 1], fname))

    classes_in_dir = sorted(
        next(os.walk(os.path.join(args.data_dir, 'train')))[1])
    classes = [a.replace('_', ' ') for a in classes_in_dir]

    assert len(classes) == len(open_ai_classes), 'num class mismatch'

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--save_dir', default='./datasets/csv')
    parser.add_argument('--data_dir', default='./datasets/data')
    parser.add_argument('--data_name', default='flowers102')
    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.data_name)

    main(args)
