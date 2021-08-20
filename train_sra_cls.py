import numpy as np
import argparse

from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import DataLoader

from pathdl.model.transformation import get_supervised_train_augmentation, get_supervised_val_augmentation
from pathdl.dataset.sampler import build_balanced_sampler
from pathdl.trainer.sra_cls_trainer import SRAClsTrainer
from pathdl.dataset.builder import dataset_selection
from pathdl.model.sra.sra import SRACls
from pathdl.utils import get_logger


def main(args):

    logger = get_logger('{}.log'.format(args.exp_name))
    logger.debug(args)

    # Define data augmentation
    transform_train = get_supervised_train_augmentation()
    transform_val = get_supervised_val_augmentation()

    # Get targets for samples
    mixed = True if args.mixed == 'yes' else False
    shuffle = True
    sampler = None

    logger.debug("Mixed: {}".format(mixed))
    logger.debug("Transform train:\n{}".format(transform_train))
    logger.debug("Transform validation:\n{}".format(transform_val))
    dataset_train, dataset_val, cls_names = dataset_selection(
        name=args.name,
        path=args.root,
        transform_train=transform_train,
        transform_val=transform_val,
        mixed=mixed,
        seed=args.seed,
    )

    # Fraction
    if args.fraction != 1.0:
        rnd = np.random.RandomState(seed=args.seed)
        n_samples = len(dataset_train)
        n_fraction = int(n_samples*args.fraction)
        id_fraction = rnd.permutation(n_samples)[:n_fraction]
        dataset_train = Subset(dataset_train, id_fraction)

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

    Examples
    --------
    In this case the code run the training of the SRA algorithm end-to-end. The first command run the unsupervised 
    training, the second the plotting of the embedding, and the third the linear classifier on top.
      
    >>> PATH="/mnt/data/dataset/tiled/kather19tiles/NCT-CRC-HE-100K/:/mnt/data/dataset/tiled/bern"
    >>> SW=0.25
    >>> SH=0.15
    >>> EXP_NAME_TRAIN="sra_sw${SW}_sh${SH}_k19:bern"
    >>> EXP_NAME_EMBED="sra_embed_sw${1}_sh${2}_k19:bern"
    >>> EXP_NAME_CLS="sra_cls_sw${1}_sh${2}_k19:bern"
    >>> MODEL_NAME_TRAIN="checkpoint_199_sra_sw${1}_sh${2}_k19:bern.pth"
    >>> 
    >>> python train_sra.py --root=$PATH --sw $SW --sh $SH --num_samples=100000 --bs=128 --exp_name $EXP_NAME_TRAIN
    >>> python embedding_sra.py --root=$PATH --name "kather19:bern" --loadpath $MODEL_NAME_TRAIN --bs 128 --exp_name $EXP_NAME_EMBED
    >>> python train_sra_cls.py --root=$PATH --name "kather19" --loadpath $MODEL_NAME_TRAIN --bs 128 --exp_name $EXP_NAME_CLS
    
    """

    parser = argparse.ArgumentParser(
        description='Train model on histological data')
    parser.add_argument('--name', type=str,
                        # default='',
                        # default='kather19',
                        # default='crctp',
                        # default='crctp+kather19',
                        # default='crctp-cstr+kather19',
                        default='crctp-clean+kather19',
                        choices=[
                            'kather19', 'crctp', 'crctp+kather19', 'crctp-cstr+kather19', 'crctp-clean+kather19'
                        ],
                        help='Name of the dataset.')
    parser.add_argument('--root', type=str,
                        # default='',
                        default='/home/abbet/Documents/lts5/phd/dataset/crctp/Training:/home/abbet/Documents/lts5/phd/dataset/kather19tiles/NCT-CRC-HE-100K',
                        help='Path to source dataset')
    parser.add_argument('--exp_name', type=str,
                        default='sra_cls',
                        help='Name of the experiment that will appear in logging.')
    parser.add_argument('--mixed', type=str,
                        default='no',
                        choices=['yes', 'no'],
                        help='Whether to generate or not mixture of tissues.')
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
                        # default="/home/abbet/checkpoint_199_sra.pth",
                        default="/home/abbet/Downloads/checkpoints/checkpoint_199_sra_sw0.5_sh0.6_k19:bern.pth",
                        help='Path to pretrained model')

    # --------- MoCo settings
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')

    args = parser.parse_args()
    main(args)
