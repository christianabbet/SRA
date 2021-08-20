from torchvision.datasets import ImageFolder
from typing import Optional
import os
import numpy as np
from PIL import Image


class BaseRemapDataset(ImageFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[object] = None,
            map: Optional[object] = None,
    ):
        """
        Creates a dataset where classes name can be remaped. Useful when training with multiple dataset with similar
        classes but different classes names.

        Parameters
        ----------
        root: str
            Path to image folder.
        transform: callable, optional
            Transformation to apply on image.
        """
        # Build folder dataset
        super(BaseRemapDataset, self).__init__(root=root, transform=transform)
        if map is not None:
            self.remap_classes(map=map)

    def remap_classes(self, map):
        """
        Remap classes according to input dictionary.

        Parameters
        ----------
        map: dict
            Remap classes dictionary.

        Examples
        --------
        Remap class Begnin and Inflamatory as NORM and LYM with 0 and 1 as new target index respectively.
        >>> remap = {'Benign': ('NORM', 0), 'Inflammatory': ('LYM', 1)}
        >>> dataset.remap_classes(map=remap)
        """

        if '*' in map:
            # Set all entries to default value
            idx_to_class = {v: "*" for k, v in self.class_to_idx.items()}
        else:
            idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Replace targets
        self.targets = [map[idx_to_class[t]][1] for t in self.targets if idx_to_class[t] in map]
        self.samples = [(s, map[idx_to_class[t]][1]) for (s, t) in self.samples if idx_to_class[t] in map]

        self.classes = [k[0] for k in map.values()]
        self.class_to_idx = {k[0]: k[1] for k in map.values()}


class HistoDataset(BaseRemapDataset):

    def __init__(
            self,
            root: str,
            set: Optional[str] = 'train',
            transform: Optional[object] = None,
            val_ratio: Optional[float] = 0.1,
            seed: Optional[int] = 0,
            map: Optional[object] = None,
    ):
        """
        Create Dataset based on the Folder

        Parameters
        ----------
        root: str
            Path to image folder.
        set: str, optional
            Set to build. Should be one of : 'train', 'val', 'full'. Default is 'train'.
        transform: callable, optional
            Transformation to apply on image.
        val_ratio: float, optional
            Ratio of validation sample with respect to training samples. Default is 0.1 (e.i., 10%).
        seed: int, optional
            Seed for splitting training and validation set. Default value is 0.

        Examples
        --------
        Create a trainset.
        >>> dataset = HistoDataset(
                root="/path/to/dataset",
                set="train"
            )
        """

        # Check if path to folder exists
        if not os.path.exists(root):
            raise FileNotFoundError

        # Build folder dataset
        super(HistoDataset, self).__init__(root=root, transform=transform, map=map)

        # Split sets is train or val sets.
        self.indices = np.random.RandomState(seed=seed).permutation(len(self.samples))
        if set == 'train':
            self.indices = self.indices[int(val_ratio*len(self.indices)):]
        elif set == 'val':
            self.indices = self.indices[:int(val_ratio*len(self.indices))]
        elif set == 'full':
            pass
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        return super(HistoDataset, self).__getitem__(self.indices[index])

    def __len__(self) -> int:
        return len(self.indices)
