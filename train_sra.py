#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import sra.builder
import sra.loader
import numpy as np
from dataset import load_dataset
from utils import AverageMeter, ProgressMeter, adjust_learning_rate, update_samples, accuracy


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Main arguments (mandatory)
parser = argparse.ArgumentParser(description='SRA evaluation process')
parser.add_argument('--src_name', type=str, default="kather19",
                    choices=["kather16", "kather19"],
                    help='Name of the source dataset')
parser.add_argument('--src_path', type=str, default="",
                    help='path to source dataset')
parser.add_argument('--tar_name', type=str, default="kather16",
                    choices=["kather16", "kather19"],
                    help='Name of the target dataset')
parser.add_argument('--tar_path', type=str, default="",
                    help='path to target dataset')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to latest checkpoint (default: none)')

# Additional arguments (optional)
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-a', '--arch', metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--checkpoint_epochs', default=50, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_samples', type=int, default=100000,
                    help='Number of samples to draw from dataset per epoch')
parser.add_argument('--sh', default=0.2, type=int, metavar='N',
                    help='Simple to hard height for step function update')
parser.add_argument('--sw', default=0.25, type=float, metavar='N',
                    help='Simple to hard width for step function update')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# MoCoV2 related-specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    default=True,
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    cudnn.benchmark = True

    print("******** Define augmentation ********")
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    print("******** Loading datasets ********")
    # Load source dataset
    _, src_dataset, _ = load_dataset(
        transforms_train=sra.loader.TwoCropsTransform(transforms.Compose(augmentation)),
        transforms_test=sra.loader.TwoCropsTransform(transforms.Compose(augmentation)),
        **{'data': args.src_path, 'dataset': args.src_name, 'seed': args.seed}
    )

    # Load target dataset
    _, tar_dataset, _ = load_dataset(
        transforms_train=sra.loader.TwoCropsTransform(transforms.Compose(augmentation)),
        transforms_test=sra.loader.TwoCropsTransform(transforms.Compose(augmentation)),
        **{'data': args.tar_path, 'dataset': args.tar_name, 'seed': args.seed}
    )

    # Attribute dataset label 0:source, 1:target
    src_dataset = update_samples(src_dataset, 0)
    tar_dataset = update_samples(tar_dataset, 1)

    # Create weighted sampler (sample the same amount of example from both sets)
    weights = np.ones(len(src_dataset)+len(tar_dataset))
    weights[:len(src_dataset)] = len(weights)/len(src_dataset)
    weights[len(src_dataset):] = len(weights)/len(tar_dataset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, args.num_samples, replacement=True)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([src_dataset, tar_dataset]),
        batch_size=args.batch_size, shuffle=False,
        sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Build model
    print("******** Building model ********")
    print("Creating model with backbone '{}'".format(args.arch))
    model = sra.builder.SRA(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    print('Use GPU {}'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    print("******** Define criterion and optimizer ********")
    filename = "checkpoint_{}+{}_sra".format(args.src_name, args.tar_name)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch+1) % args.checkpoint_epochs == 0:
            print('Saving model epoch: {}'.format(epoch))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f='{}_{:04d}.pth.tar'.format(filename, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_ind = AverageMeter('LossIND', ':.4e')
    losses_crd = AverageMeter('LossCRD', ':.4e')
    top1_d0 = AverageMeter('AccD0@1', ':6.2f')
    top1_d1 = AverageMeter('AccD1@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, losses_ind, losses_crd, top1_d0, top1_d1],
        prefix="Epoch: [{}]".format(epoch))

    # Compute simple to hard ratio
    s2h_topk_r = np.floor(epoch/(args.sw*args.epochs))*args.sh
    print('Consider top: {}%'.format(s2h_topk_r*100))

    for i, (images, d_set) in enumerate(train_loader):
        # images[0]: key, image[1]: query, d_set: label source (0) or target (1)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            d_set = d_set.cuda(args.gpu, non_blocking=True)

        # compute output
        l_ind_d0, t_ind_d0, l_ind_d1, t_ind_d1, h_crd_d0tod1, h_crd_d1tod0 = model(im_q=images[0], im_k=images[1], dset=d_set)

        # 1. In-domain Self-supervision
        loss_ind_d0 = criterion(l_ind_d0, t_ind_d0)
        loss_ind_d1 = criterion(l_ind_d1, t_ind_d1)
        loss_ind = (1/args.batch_size)*(loss_ind_d0 + loss_ind_d1)

        # 2. Cross-domain self-supervision
        h_crd_d0tod1 = h_crd_d0tod1[h_crd_d0tod1.argsort()[:int(s2h_topk_r*len(h_crd_d0tod1))]]
        h_crd_d1tod0 = h_crd_d1tod0[h_crd_d1tod0.argsort()[:int(s2h_topk_r*len(h_crd_d1tod0))]]
        loss_crd_d0tod1 = h_crd_d0tod1.sum(dim=0)
        loss_crd_d1tod0 = h_crd_d1tod0.sum(dim=0)
        loss_crd = (1 / (args.batch_size*s2h_topk_r)) * (loss_crd_d0tod1 + loss_crd_d1tod0)

        # 3. Overall loss
        loss = loss_ind + loss_crd

        # 4. Update metrics
        acc1d0, acc5d0 = accuracy(l_ind_d0, t_ind_d0, topk=(1, 5))
        acc1d1, acc5d1 = accuracy(l_ind_d1, t_ind_d1, topk=(1, 5))
        losses_ind.update(loss_ind.item(), images[0].size(0))
        losses_crd.update(loss_crd.item(), images[0].size(0))
        losses.update(loss.item(), images[0].size(0))
        top1_d0.update(acc1d0[0], images[0].size(0))
        top1_d1.update(acc1d1[0], images[0].size(0))

        # 5. Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()
