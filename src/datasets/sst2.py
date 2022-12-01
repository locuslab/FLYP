import glob
import os

import numpy as np
import torch
import torchvision
from PIL import Image


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
        for i, c in enumerate(self.class_list):
            rand_idx = np.random.randint(0, self.class_len_list[i])
            img_name = self.img_list[i][rand_idx]
            image = self.transforms(Image.open(img_name))
            batch_img.append(image)

        batch_img = torch.stack(batch_img, dim=0)

        return batch_img


class sst2:
    test_subset = None

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 subset='test',
                 classnames=None,
                 custom=False,
                 k=None,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k
        if self.k is not None:
            self.train_location = os.path.join(location, 'sst2',
                                               f'train_shot_{self.k}')
        else:
            self.train_location = os.path.join(location, 'sst2', 'train')

        print("Loading Train Data from ", self.train_location)
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

        self.test_location = os.path.join(location, 'sst2', self.test_subset)
        print("Loading Test Data from ", self.test_location)
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=self.test_location, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        self.classnames = [
            'negative',
            'positive',
        ]


class sst2Val(sst2):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'val'
        super().__init__(*args, **kwargs)


class sst2Test(sst2):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)