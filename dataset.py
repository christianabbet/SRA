import torchvision.datasets as datasets
import os
import numpy as np
from torch.utils.data import Subset


def load_dataset(transforms_train, transforms_test, **args):

    print("******** Building dataset ********\n\tname: {}\n\tpath: {}\n\tseed: {}".format(
        args.get('dataset', ''), args.get('data', ''), args.get('seed', '')))

    if args.get('dataset', '') == "kather19":
        return load_kather19(transforms_train, transforms_test, **args)
    elif args.get('dataset', '') == "kather16":
        return load_kather16(transforms_train, transforms_test, **args)
    else:
        raise NotImplementedError("Unknown dataset")


def load_kather19(transforms_train, transforms_test, **args):
    traindir = os.path.join(args.get('data', ''), 'NCT-CRC-HE-100K')
    testdir = os.path.join(args.get('data', ''), 'CRC-VAL-HE-7K')

    trainval_dataset = datasets.ImageFolder(
        traindir, transforms_train,
        is_valid_file=lambda f: f.endswith(".jpeg")
    )

    test_dataset = datasets.ImageFolder(
        testdir, transforms_test,
        is_valid_file=lambda f: f.endswith(".jpeg")
    )

    return test_dataset.class_to_idx, trainval_dataset, test_dataset


def load_kather16(transforms_train, transforms_test, ratio=0.3, **args):

    trainval_dataset = datasets.ImageFolder(
        args.get('data', ''), transform=transforms_train,
    )
    test_dataset = datasets.ImageFolder(
        args.get('data', ''), transform=transforms_test,
    )

    rnd = np.random.RandomState(seed=args.get('seed', 0))
    id_shuffle = rnd.permutation(len(trainval_dataset))
    id_train = id_shuffle[int(len(id_shuffle)*ratio):]
    id_test = id_shuffle[:int(len(id_shuffle)*ratio)]

    return test_dataset.class_to_idx, Subset(trainval_dataset, id_train), Subset(test_dataset, id_test)
