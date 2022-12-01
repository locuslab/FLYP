import os
import torch
import torchvision
import wilds

from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import numpy as np
from PIL import Image
import glob


class CustomDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transforms = transform
        self.class_list = sorted(os.listdir(root))
        self.img_list = []
        self.class_len_list = []
        for i, c in enumerate(self.class_list):
            root_child = os.path.join(root, c)
            self.img_list.append(sorted(glob.glob(root_child + "/*")))
            self.class_len_list.append(len(self.img_list[-1]))

    def __len__(self):
        total_len = 0
        for i, c in enumerate(self.class_list):
            total_len += len(self.img_list[i])
        return total_len

    def __getitem__(self, idx):
        batch_img = []
        # batch_y = []
        for i, c in enumerate(self.class_list):
            rand_idx = np.random.randint(0, self.class_len_list[i])
            img_name = self.img_list[i][rand_idx]
            image = self.transforms(Image.open(img_name))
            batch_img.append(image)
            # batch_y.append(i)

        batch_img = torch.stack(batch_img, dim=0)
        # batch_y = torch.stack(batch_y, dim=0)
        # if self.transforms:
        #     batch_img = self.transforms(batch_img)

        return batch_img


class Flowers102:
    test_subset = None

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 subset='test',
                 classnames=None,
                 custom=False,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_location = os.path.join(location, 'flowers102', 'train')
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_location, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)
        if custom:
            self.train_dataset_custom = CustomDataset(root=self.train_location,
                                                      transform=preprocess)
            self.train_loader_custom = torch.utils.data.DataLoader(
                self.train_dataset_custom,
                batch_size=1,
                shuffle=True,
                num_workers=self.num_workers)

        self.test_location = os.path.join(location, 'flowers102',
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
            'air plant', 'alpine sea holly', 'anthurium', 'artichoke',
            'azalea', 'balloon flower', 'barbeton daisy', 'bearded iris',
            'bee balm', 'bird of paradise', 'bishop of llandaff',
            'black-eyed susan', 'blackberry lily', 'blanket flower',
            'bolero deep blue', 'bougainvillea', 'bromelia', 'buttercup',
            'californian poppy', 'camellia', 'canna lily', 'canterbury bells',
            'cape flower', 'carnation', 'cautleya spicata', 'clematis',
            "colt's foot", 'columbine', 'common dandelion', 'corn poppy',
            'cyclamen', 'daffodil', 'desert-rose', 'english marigold',
            'fire lily', 'foxglove', 'frangipani', 'fritillary',
            'garden phlox', 'gaura', 'gazania', 'geranium',
            'giant white arum lily', 'globe flower', 'globe thistle',
            'grape hyacinth', 'great masterwort', 'hard-leaved pocket orchid',
            'hibiscus', 'hippeastrum', 'japanese anemone', 'king protea',
            'lenten rose', 'lotus', 'love in the mist', 'magnolia', 'mallow',
            'marigold', 'mexican aster', 'mexican petunia', 'monkshood',
            'moon orchid', 'morning glory', 'orange dahlia', 'osteospermum',
            'oxeye daisy', 'passion flower', 'pelargonium', 'peruvian lily',
            'petunia', 'pincushion flower', 'pink and yellow dahlia',
            'pink primrose', 'poinsettia', 'primula',
            'prince of wales feathers', 'purple coneflower', 'red ginger',
            'rose', 'ruby-lipped cattleya', 'siam tulip', 'silverbush',
            'snapdragon', 'spear thistle', 'spring crocus', 'stemless gentian',
            'sunflower', 'sweet pea', 'sweet william', 'sword lily',
            'thorn apple', 'tiger lily', 'toad lily', 'tree mallow',
            'tree poppy', 'trumpet creeper', 'wallflower', 'water lily',
            'watercress', 'wild pansy', 'windflower', 'yellow iris'
        ]


class Flowers102Val(Flowers102):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'val'
        super().__init__(*args, **kwargs)


class Flowers102Test(Flowers102):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)