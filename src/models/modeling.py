import torch
import copy

import clip.clip as clip

from src.models import utils
import open_clip


class CLIPEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()
        if args.model == 'ViT-L-14':
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained='laion400m_e31')
        elif args.model == 'ViT-B-16':
            print("****************Loading ViTB16 from openCLIP****************")
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained='laion400m_e31')
        else:
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                args.model, args.device, jit=False)
        self.cache_dir = args.cache_dir

    def forward(self, images, text=None):
        assert self.model is not None
        # if text==None:
        #     return self.model.encode_image(images)
        # else:
        return self.model(images, text)

    def save(self, filename):
        print(f'Saving clip encoder to {filename}')
        utils.torch_save(self, filename)
        # torch.save(self.model, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading image encoder from {filename}')
        if logger != None:
            logger.info(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, shape=[512, 1000]):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())

        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading classification head from {filename}')
        if logger != None:
            logger.info(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class ImageClassifier_Norm(torch.nn.Module):
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)