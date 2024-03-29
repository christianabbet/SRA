from dataset.base import BaseRemapDataset, HistoDataset
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from typing import Tuple, Optional, Iterable
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from dataset.constants import (const_kather19, const_crctp, const_crctp_to_kather19, const_crctp_cstr_to_kather19)


def dataset_selection(
        name: str, path: str, transform_train: object, transform_val: object, **kwargs,
) -> Tuple[Dataset, Dataset, Iterable[str]]:
    """
    Select and build dataset for training.

    Parameters
    ----------
    name: str
        Name of the dataset. Should be one of
        'kather19'            : 9 classes with [ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM]
        'crctp+kather19'      : 10 classes with [ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM, CSTR]
        'custom'              : Your classes
    path: str
        Path tot the dataset. When feeding two dataset path should be composed as 'path1:path2', where path1 is the
        path to the first dataset and path2 the path to the second one.
    transform_train: object
        Transform to apply on train images.
    transform_val: object
        Transform to apply on validation images.

    Returns
    -------
    dataset_train: Dataset
        Training dataset.
    dataset_val: Dataset
        Validation dataset.
    cls_labels: list of str
        Ordered name of the classes from 0 to C.
    """

    # Build dataset on Kather19
    if name == 'kather19':
        return build_dataset(path=path, transform_train=transform_train, transform_val=transform_val,
                             remap=const_kather19, **kwargs)

    # Build dataset on CRCTP
    elif name == 'crctp':
        return build_dataset(path=path, transform_train=transform_train, transform_val=transform_val,
                             remap=const_crctp, **kwargs)

    # Build dataset on Kather19 and CRCTP
    elif name == 'crctp+kather19' or name == 'crctp-cstr+kather19':
        paths = path.split(':')

        if name == 'crctp-cstr+kather19':
            remap = const_crctp_cstr_to_kather19
        else:
            remap = const_crctp_to_kather19

        # CRCTP
        d_train_1, d_val_1, _ = build_dataset(
            path=paths[0],
            transform_train=transform_train, transform_val=transform_val,
            remap=remap, **kwargs
        )
        # Kather19
        d_train_2, d_val_2, _ = build_dataset(
            path=paths[1],
            transform_train=transform_train, transform_val=transform_val,
            remap=None, **kwargs
        )

        cls_labels = merge_classes(d_train_1.class_to_idx, d_train_2.class_to_idx)
        return ConcatDataset((d_train_1, d_train_2)), ConcatDataset((d_val_1, d_val_2)), list(cls_labels.keys())

    elif name == 'custom':
        return build_dataset(path=path, transform_train=transform_train, transform_val=transform_val,
                             remap=None, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset type: {}'.format(name))


def merge_classes(cls_map_1: dict, cls_map_2: dict) -> dict:
    """
    Merge dictionary of different dataset to create a single class list.

    Parameters
    ----------
    cls_map_1: dict
        Dictionary of the first dataset (e.g. : {'ADI': 0, 'TUM': 1}
    cls_map_2: dict
        Dictionary of the second dataset (e.g. : {'ADI': 0, 'LYM': 2}
    Returns
    -------
    cls_labels: dict
        Merge labels (e.g.  {'ADI': 0, 'TUM': 1, 'LYM': 2})
    """

    cls_merged = cls_map_1.copy()
    cls_merged.update(cls_map_2)
    keys = list(cls_merged.keys())
    values = list(cls_merged.values())
    cls_merged = {keys[id_]: values[id_] for id_ in np.argsort(list(cls_merged.values()))}
    return cls_merged
    # n_cls = np.hstack((list(cls_map_1.values()), list(cls_map_2.values()))).max() + 1
    # cls_labels = ['']*n_cls
    #
    # r_cls_map_1 = {v: k for k, v in cls_map_1.items()}
    # r_cls_map_2 = {v: k for k, v in cls_map_2.items()}
    #
    # for i in range(n_cls):
    #     cls_labels[i] = r_cls_map_1.get(i, cls_labels[i])
    #     cls_labels[i] = r_cls_map_2.get(i, cls_labels[i])
    #
    # return dict(zip(cls_labels, np.arange(cls_labels.__len__())))


def build_dataset(
        path: str, transform_train: object, transform_val: object, remap: Optional[dict] = None, **kwargs
) -> Tuple[BaseRemapDataset, BaseRemapDataset, Iterable[str]]:
    """
    Build train and validation dataset with transforms. Classes can be remaped with the remap argument. Mixed is used
    to generate mixture of tissues.

    Parameters
    ----------
    path: str
        Path tot the dataset. When feeding two dataset path should be composed as 'path1:path2', where path1 is the
        path to the first dataset and path2 the path to the second one.
    transform_train: object
        Transform to apply on train images.
    transform_val: object
        Transform to apply on validation images.
    remap: dict
        Class mapping to update. (e.g. to replace 'Adipose' with 'ADI' and class index 0, {'Adipose': ('ADI', 0)}

    Returns
    -------
    dataset_train: BaseRemapDataset
        Training dataset.
    dataset_val: BaseRemapDataset
        Validation dataset.
    cls_labels: list of str
        Ordered name of the classes from 0 to C.
    """

    Cls = HistoDataset

    dataset_train = Cls(root=path, transform=transform_train, set='train', map=remap, **kwargs)
    dataset_val = Cls(root=path, transform=transform_val, set='val', map=remap, **kwargs)
    # if remap is not None:
    #     dataset_train.remap_classes(remap)
    #     dataset_val.remap_classes(remap)
    return dataset_train, dataset_val, dataset_val.classes


def build_balanced_sampler(targets) -> WeightedRandomSampler:
    """
    Build a balanced sampler for ImageFolder dataset. The probability is scaled as a function of the occurence of the
    class. For examples if class 0 has 100 samples and class 1 has 900 samples, the weigths would respectively be
    1000/100 = 10 and 1000/900 = 1.11.
    Here, the replacement if set to Fals, hence the total number of sample is based on the class with the lowest number
    of examples (2 classes, 100 samples -> 200 samples).

    Parameters
    ----------
    targets: ImageFolder
        Dataset to balance sample from.
    Returns
    -------
    sampler: WeightedRandomSampler
        Sampler to feed to the data loader.
    """
    cls, n_cls = np.unique(targets, return_counts=True)
    n_weights = dict(zip(cls, np.sum(n_cls) / n_cls))

    return WeightedRandomSampler(
        weights=[n_weights[t] for t in targets],
        num_samples=int(len(n_cls) * int(n_cls.mean())),
        replacement=True,
    )
