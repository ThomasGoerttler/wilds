import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import torch

def initialize_transform(transform_name, config, dataset):
    if transform_name is None:
        return None
    elif transform_name=='bert':
        return initialize_bert_transform(config)
    elif transform_name=='image_base':
        return initialize_image_base_transform(config, dataset)
    elif transform_name=='image_resize_and_center_crop':
        return initialize_image_resize_and_center_crop_transform(config, dataset)
    elif transform_name=='poverty_train':
        return initialize_poverty_train_transform()
    elif transform_name=='rxrx1':
        return initialize_rxrx1_transform(dataset)
    else:
        raise ValueError(f"{transform_name} not recognized")

def initialize_bert_transform(config):
    assert 'bert' in config.model
    assert config.max_token_length is not None

    tokenizer = getBertTokenizer(config.model)
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.max_token_length,
            return_tensors='pt')
        if config.model == 'bert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif config.model == 'distilbert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

def getBertTokenizer(model):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer

def initialize_image_base_transform(config, dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(dataset.original_resolution)!=max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    if config.target_resolution is not None and config.dataset!='fmow':
        transform_steps.append(transforms.Resize(config.target_resolution))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform = transforms.Compose(transform_steps)
    return transform

def initialize_image_resize_and_center_crop_transform(config, dataset):
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(int(res*config.resize_scale) for res in dataset.original_resolution)
    if config.target_resolution is not None:
        target_resolution = config.target_resolution
    else:
        target_resolution = dataset.original_resolution
    transform = transforms.Compose([
        transforms.Resize(scaled_resolution),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def initialize_poverty_train_transform():
    transforms_ls = [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
        transforms.ToTensor()]
    rgb_transform = transforms.Compose(transforms_ls)

    def transform_rgb(img):
        # bgr to rgb and back to bgr
        img[:3] = rgb_transform(img[:3][[2,1,0]])[[2,1,0]]
        return img
    transform = transforms.Lambda(lambda x: transform_rgb(x))
    return transform


def initialize_rxrx1_transform(dataset: str):

    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    def random_d8(x: torch.Tensor) -> torch.Tensor:
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            x = TF.rotate(x, angle)
        if random.random() < 0.5:
            x = TF.hflip(x)
        return x
    t_random_d8 = transforms.Lambda(lambda x: random_d8(x))

    if dataset == 'train':
        transforms_ls = [
            t_random_d8,
            transforms.ToTensor(),
            t_standardize,
        ]
    elif dataset == 'test':
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)

    return transform
