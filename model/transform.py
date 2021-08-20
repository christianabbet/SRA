from typing import Optional
from PIL import Image
import numpy as np
from torchvision import transforms
from PIL import ImageFilter
import random


from albumentations import (RandomRotate90, GridDistortion, HueSaturationValue, ISONoise, GaussNoise,
                            RandomGamma, RandomBrightnessContrast)


def get_supervised_train_augmentation(normalize: Optional[bool] = True, heavy: Optional[bool] = True):
    """
    Build standard training transformation for Histology networks. It includes, resizing, color jittering, horzonal
    and vertical flipping, gaussian blurring, and normalization.

    Parameters
    ----------
    normalize: bool, optional
        Whether to apply or not normalization to the data.
    heavy: bool, optional
        Whether to apply heavy transformations

    Returns
    -------
    tfm: object
        Composition of transformation for the training set.

    """

    if heavy:
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            PILtoNumpy(),
            ApplyOnKey(func=RandomRotate90(p=1.0), key='image'),
            # ApplyOnKey(func=GaussianBlur(p=0.3, blur_limit=5), key='image'),
            ApplyOnKey(func=RandomGamma(p=0.3, gamma_limit=(80, 120)), key='image'),
            # ApplyOnKey(func=RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2), key='image'),
            # ApplyOnKey(func=HueSaturationValue(p=0.3, hue_shift_limit=20, sat_shift_limit=10, val_shift_limit=10),
            #            key='image'),
            ApplyOnKey(func=GridDistortion(p=0.3, num_steps=5, distort_limit=(-0.3, 0.3)), key='image'),
            ApplyOnKey(func=ISONoise(p=0.3, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)), key='image'),
            ApplyOnKey(func=GaussNoise(p=0.3, var_limit=(10.0, 50.0)), key='image'),
            NumpyToPIL(),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
            transforms.RandomApply([
                # StainJitter(method='macenko', sigma1=0.1, sigma2=0.1, augment_background=True, n_thread=1),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.742, 0.616, 0.731], std=[0.211, 0.274, 0.202]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Remove normalization if False
    if not normalize:
        tfm.transforms = tfm.transforms[:-1]
    return tfm


def get_supervised_val_augmentation(normalize=True):
    """
    Build standard validation / testing transformation for Histology networks. It includes, resizing, and normalization.

    Parameters
    ----------
    normalize: bool, optional
        Whether to apply or not normalization to the data.

    Returns
    -------
    tfm: object
        Composition of transformation for the testing set.

    """

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.742, 0.616, 0.731], std=[0.211, 0.274, 0.202]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Remove normalization if False
    if not normalize:
        tfm.transforms = tfm.transforms[:-1]
    return tfm


class ApplyOnKey:

    def __init__(self, func, key):
        self.func = func
        self.key = key

    def __call__(self, x):
        data = {self.key: x}
        return self.func(**data)[self.key]

    def __repr__(self):
        return self.func.__repr__()


class PILtoNumpy:
    def __call__(self, x):
        return np.array(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NumpyToPIL:
    def __call__(self, x):
        return Image.fromarray(x.astype(np.uint8))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)