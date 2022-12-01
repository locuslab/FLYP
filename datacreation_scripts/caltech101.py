import os
import pandas as pd
import argparse

classes = [
    'off-center face', 'centered face', 'leopard', 'motorbike', 'accordion',
    'airplane', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular',
    'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera',
    'cannon', 'side of a car', 'ceiling fan', 'cellphone', 'chair',
    'chandelier', 'body of a cougar cat', 'face of a cougar cat', 'crab',
    'crayfish', 'crocodile', 'head of a  crocodile', 'cup', 'dalmatian',
    'dollar bill', 'dolphin', 'dragonfly', 'electric guitar', 'elephant',
    'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'head of a flamingo',
    'garfield', 'gerenuk', 'gramophone', 'grand piano', 'hawksbill',
    'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline skate',
    'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster',
    'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret',
    'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza',
    'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
    'schooner', 'scissors', 'scorpion', 'sea horse', 'snoopy (cartoon beagle)',
    'soccer ball', 'stapler', 'starfish', 'stegosaurus', 'stop sign',
    'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch',
    'water lilly', 'wheelchair', 'wild cat', 'windsor chair', 'wrench',
    'yin and yang symbol'
]

templates = [
    lambda c: f'a photo of a {c}.', lambda c: f'a painting of a {c}.',
    lambda c: f'a plastic {c}.', lambda c: f'a sculpture of a {c}.',
    lambda c: f'a sketch of a {c}.', lambda c: f'a tattoo of a {c}.',
    lambda c: f'a toy {c}.', lambda c: f'a rendition of a {c}.',
    lambda c: f'a embroidered {c}.', lambda c: f'a cartoon {c}.',
    lambda c: f'a {c} in a video game.', lambda c: f'a plushie {c}.',
    lambda c: f'a origami {c}.', lambda c: f'art of a {c}.',
    lambda c: f'graffiti of a {c}.', lambda c: f'a drawing of a {c}.',
    lambda c: f'a doodle of a {c}.', lambda c: f'a photo of the {c}.',
    lambda c: f'a painting of the {c}.', lambda c: f'the plastic {c}.',
    lambda c: f'a sculpture of the {c}.', lambda c: f'a sketch of the {c}.',
    lambda c: f'a tattoo of the {c}.', lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.', lambda c: f'the embroidered {c}.',
    lambda c: f'the cartoon {c}.', lambda c: f'the {c} in a video game.',
    lambda c: f'the plushie {c}.', lambda c: f'the origami {c}.',
    lambda c: f'art of the {c}.', lambda c: f'graffiti of the {c}.',
    lambda c: f'a drawing of the {c}.', lambda c: f'a doodle of the {c}.'
]


def main(args):

    assert len(classes) == 101, 'number of classes are less'
    print(args.data_dir)
    classes_in_dir = sorted(next(os.walk(args.data_dir))[1])

    assert len(classes_in_dir) == len(classes), 'number of classes mismatch'

    with open(args.save_file, 'w') as f:
        f.write('title\tfilepath\n')
        for i, dir_name in enumerate(classes_in_dir):
            for file in os.listdir(os.path.join(args.data_dir, dir_name)):
                assert 'jpg' in file or 'jpeg' in file, 'extension mismatch'
                full_path = os.path.join(args.data_dir, dir_name, file)
                for template in templates:
                    f.write(f'{template(classes[i])}\t{full_path}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--type',
                        choices=['train', 'test', 'val'],
                        default='train')
    parser.add_argument('--save_dir', default='./datasets/csv')
    parser.add_argument('--data_dir', default='./datasets/data')
    parser.add_argument('--data_name', default='caltech-101')
    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.data_name, args.type)
    args.save_file = os.path.join(args.save_dir, args.data_name,
                                  f'{args.type}.csv')
    os.makedirs(os.path.join(args.save_dir, args.data_name), exist_ok=True)
    main(args)
