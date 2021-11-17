import argparse
import numpy as np

from model.utils import get_logger
from model.transform import get_supervised_train_augmentation, get_supervised_val_augmentation, TwoCropsTransform
from model.sra import SRA
from model.sra_trainer import SRATrainer
from dataset.builder import build_dataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, ConcatDataset


def main(args):

    logger = get_logger('{}.log'.format(args.exp_name))
    logger.debug(args)

    # Define data augmentation
    transform_train = get_supervised_train_augmentation(heavy=True)
    transform_val = get_supervised_val_augmentation()

    logger.debug("Transform train:\n{}".format(transform_train))
    logger.debug("Transform validation:\n{}".format(transform_val))

    dataset_paths = args.root.split(':')

    use_lind = bool(args.lind == 'yes')
    use_lcrd = bool(args.lcrd == 'yes')
    use_e2h = bool(args.e2h == 'yes')
    use_me = bool(args.me == 'yes')

    dataset_train = []
    dataset_val = []
    for i, p in enumerate(dataset_paths):
        # Create dataset
        logger.debug("Load dataset {} from: {}".format(i, p))
        dataset_train_, dataset_val_, _ = build_dataset(
            path=p,
            transform_train=TwoCropsTransform(transform_train),
            transform_val=TwoCropsTransform(transform_val),
        )
        # Remap dataset labels to current dataset index
        new_classes = {c: ('D{}'.format(i), i) for c in dataset_train_.classes}
        dataset_train_.remap_classes(new_classes)
        dataset_val_.remap_classes(new_classes)
        # Limit size of dataset to the expected number of samples
        dataset_train.append(dataset_train_)
        dataset_val.append(dataset_val_)

    # Create weighted sampler (sample the same amount of example from both sets)
    ratio_samples = [int(r) for r in args.ratio_samples.split(':')]
    weights = np.concatenate([(r/len(d))*np.ones(len(d)) for r, d in zip(ratio_samples, dataset_train)])
    weights *= len(weights)
    sampler = WeightedRandomSampler(weights, args.num_samples, replacement=True)

    loader_train = DataLoader(
        dataset=ConcatDataset(dataset_train), batch_size=args.bs, num_workers=args.j, sampler=sampler, drop_last=True)
    loader_val = DataLoader(
        dataset=ConcatDataset(dataset_val), batch_size=args.bs, num_workers=args.j, shuffle=True, drop_last=True)

    # Define model and train it
    model = SRA(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        n_dataset=len(dataset_train),
        device=args.device,
        mean_entropy=use_me,
    )

    trainer = SRATrainer(
        model=model,
        train_loader=loader_train,
        val_loader=loader_val,
        use_lind=use_lind,
        use_lcrd=use_lcrd,
        use_e2h=use_e2h,
        opt_lr=args.lr,
        opt_momentum=args.momentum,
        opt_weight_decay=args.wd,
        t_max=args.epochs,
        sh=args.sh,
        sw=args.sw,
        checkpoint_epochs=args.checkpoint_epochs,
        device=args.device,
        prefix=args.exp_name,
        logger=logger,
    )
    trainer.train(n_epochs=args.epochs)


if __name__ == '__main__':
    """
    Train SRA model 
    """

    parser = argparse.ArgumentParser(
        description='Train model on histological data')
    parser.add_argument('--root', type=str,
                        default='',
                        help='Path to source dataset')
    parser.add_argument('--exp_name', type=str,
                        default='sra',
                        help='Name of the experiment that will appear in logging.')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples to draw from dataset per epoch')
    parser.add_argument('--ratio_samples', type=str,
                        default="1:1:2",
                        help='Samples ratio')

    # --------- Trainer settings
    parser.add_argument('--device', default="cuda", type=str,
                        help='Either cpu or cuda.')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.03, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum of SGD solver')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--j', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--bs', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--sh', default=0.15, type=float, metavar='N',
                        help='Simple to hard height for step function update')
    parser.add_argument('--sw', default=0.25, type=float, metavar='N',
                        help='Simple to hard width for step function update')
    parser.add_argument('--checkpoint_epochs', default=50, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lind', type=str,
                        default='yes',
                        choices=['yes', 'no'],
                        help='Use in-domain learning procedure')
    parser.add_argument('--lcrd', type=str,
                        default='yes',
                        choices=['yes', 'no'],
                        help='Use cross-domain learning procedure')
    parser.add_argument('--e2h', type=str,
                        default='yes',
                        choices=['yes', 'no'],
                        help='Use easy-to-hard learning procedure')
    parser.add_argument('--me', type=str,
                        default='yes',
                        choices=['yes', 'no'],
                        help='Compute mean entropy.')

    # --------- MoCo settings
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.2, type=float,
                        help='softmax temperature (default: 0.2)')

    args = parser.parse_args()

    main(args)
