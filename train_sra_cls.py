import numpy as np
import argparse

from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import DataLoader

from model.transform import get_supervised_train_augmentation, get_supervised_val_augmentation
from dataset.builder import dataset_selection
from model.sra import SRACls
from model.utils import get_logger
from dataset.builder import build_balanced_sampler
from model.sra_cls_trainer import SRAClsTrainer


def main(args):

    logger = get_logger('{}.log'.format(args.exp_name))
    logger.debug(args)

    # Define data augmentation
    transform_train = get_supervised_train_augmentation()
    transform_val = get_supervised_val_augmentation()

    # Get targets for samples
    shuffle = True
    sampler = None

    logger.debug("Transform train:\n{}".format(transform_train))
    logger.debug("Transform validation:\n{}".format(transform_val))
    dataset_train, dataset_val, cls_names = dataset_selection(
        name=args.name,
        path=args.root,
        transform_train=transform_train,
        transform_val=transform_val,
        seed=args.seed,
    )

    # Fraction
    if args.fraction != 1.0:
        rnd = np.random.RandomState(seed=args.seed)
        # Subset train
        n_samples_train = len(dataset_train)
        n_fraction_train = int(n_samples_train*args.fraction)
        dataset_train = Subset(dataset_train, rnd.permutation(n_samples_train)[:n_fraction_train])
        # Subset val
        n_samples_val = len(dataset_val)
        n_fraction_val = int(n_samples_val*args.fraction)
        dataset_val = Subset(dataset_val, rnd.permutation(n_samples_val)[:n_fraction_val])

    # Load train and validation dataset
    logger.debug("Balanced classes: {}".format(args.balance))
    if args.balance == 'yes':
        # Look at target distribution to create balanced sampler
        if isinstance(dataset_train, ConcatDataset):
            targets = np.hstack([np.array(d.targets)[d.indices] for d in dataset_train.datasets])
            # Create sampler based on targets
            sampler = build_balanced_sampler(targets=targets)
            shuffle = False
        elif isinstance(dataset_train, Subset):
            logger.debug("Not implemented fraction + balanced, will not use balanced samples")
        else:
            targets = np.array(dataset_train.targets)[dataset_train.indices]
            # Create sampler based on targets
            sampler = build_balanced_sampler(targets=targets)
            shuffle = False

    loader_train = DataLoader(dataset=dataset_train, batch_size=args.bs, num_workers=args.j,
                              shuffle=shuffle, sampler=sampler)
    loader_val = DataLoader(dataset=dataset_val, batch_size=args.bs, num_workers=args.j,
                            shuffle=False)

    # Define model and train it
    model = SRACls(
        n_cls=len(cls_names),
        dim=args.moco_dim,
        device=args.device,
    )

    trainer = SRAClsTrainer(
        model=model,
        train_loader=loader_train,
        val_loader=loader_val,
        opt_lr=args.lr,
        t_max=args.epochs,
        device=args.device,
        prefix=args.exp_name,
        logger=logger,
        loadpath=args.loadpath,
        name_classes=cls_names,
    )

    trainer.train(n_epochs=args.epochs)


if __name__ == '__main__':
    """
    Train SRA classification model.
    """

    parser = argparse.ArgumentParser(
        description='Train model on histological data')
    parser.add_argument('--name', type=str,
                        default='',
                        choices=[
                            'kather19', 'crctp-cstr+kather19', 'custom'
                        ],
                        help='Name of the dataset.')
    parser.add_argument('--root', type=str,
                        default='',
                        # default='/home/abbet/Documents/lts5/phd/dataset/crctp/Training:/home/abbet/Documents/lts5/phd/dataset/kather19tiles/NCT-CRC-HE-100K',
                        help='Path to source dataset')
    parser.add_argument('--exp_name', type=str,
                        default='sra_cls',
                        help='Name of the experiment that will appear in logging.')
    parser.add_argument('--balance', type=str,
                        default='yes',
                        choices=['yes', 'no'],
                        help='Name of the experiment that will appear in logging.')

    # --------- Trainer settings
    parser.add_argument('--device', default="cuda", type=str,
                        help='Either cpu or cuda.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for data loaders')
    parser.add_argument('--fraction', default=0.01, type=float,
                        help='Percentage of labels')
    parser.add_argument('--j', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--bs', default=128, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--loadpath', type=str,
                        default="",
                        # default="/home/abbet/checkpoint_199_sra.pth",
                        # default="/home/abbet/Downloads/checkpoints/checkpoint_199_sra_sw0.5_sh0.6_k19:bern.pth",
                        help='Path to pretrained model')

    # --------- MoCo settings
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')

    args = parser.parse_args()
    main(args)
