import os
import torch
import torchvision
import wilds

from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class Caltech101:
    test_subset = None

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 subset='test',
                 classnames=None,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_location = os.path.join(location, 'caltech-101', 'train')
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_location, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        self.test_location = os.path.join(location, 'caltech-101',
                                          self.test_subset)
        print("Loading Test Data from ", self.test_location)
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=self.test_location, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        self.classnames = [
            'off-center face',
            'centered face',
            'leopard',
            'motorbike',
            'accordion',
            'airplane',
            'anchor',
            'ant',
            'barrel',
            'bass',
            'beaver',
            'binocular',
            'bonsai',
            'brain',
            'brontosaurus',
            'buddha',
            'butterfly',
            'camera',
            'cannon',
            'side of a car',
            'ceiling fan',
            'cellphone',
            'chair',
            'chandelier',
            'body of a cougar cat',
            'face of a cougar cat',
            'crab',
            'crayfish',
            'crocodile',
            'head of a  crocodile',
            'cup',
            'dalmatian',
            'dollar bill',
            'dolphin',
            'dragonfly',
            'electric guitar',
            'elephant',
            'emu',
            'euphonium',
            'ewer',
            'ferry',
            'flamingo',
            'head of a flamingo',
            'garfield',
            'gerenuk',
            'gramophone',
            'grand piano',
            'hawksbill',
            'headphone',
            'hedgehog',
            'helicopter',
            'ibis',
            'inline skate',
            'joshua tree',
            'kangaroo',
            'ketch',
            'lamp',
            'laptop',
            'llama',
            'lobster',
            'lotus',
            'mandolin',
            'mayfly',
            'menorah',
            'metronome',
            'minaret',
            'nautilus',
            'octopus',
            'okapi',
            'pagoda',
            'panda',
            'pigeon',
            'pizza',
            'platypus',
            'pyramid',
            'revolver',
            'rhino',
            'rooster',
            'saxophone',
            'schooner',
            'scissors',
            'scorpion',
            'sea horse',
            'snoopy (cartoon beagle)',
            'soccer ball',
            'stapler',
            'starfish',
            'stegosaurus',
            'stop sign',
            'strawberry',
            'sunflower',
            'tick',
            'trilobite',
            'umbrella',
            'watch',
            'water lilly',
            'wheelchair',
            'wild cat',
            'windsor chair',
            'wrench',
            'yin and yang symbol'
        ]


class Caltech101Val(Caltech101):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'val'
        super().__init__(*args, **kwargs)


class Caltech101Test(Caltech101):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)