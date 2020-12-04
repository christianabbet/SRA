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
from torch.utils.tensorboard import SummaryWriter
from datetime import date
import os

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
                    help='path to source dataset. For Kather19, root folder should contain CRC-VAL-HE-7K (test) and'
                         ' NCT-CRC-HE-100K (train/val). For Kather16, should contain class folders')
parser.add_argument('--tar_name', type=str, default="kather16",
                    choices=["kather16", "kather19"],
                    help='Name of the target dataset')
parser.add_argument('--tar_path', type=str, default="",
                    help='path to source dataset. For Kather19, root folder should contain CRC-VAL-HE-7K (test) and'
                         ' NCT-CRC-HE-100K (train/val). For Kather16, should contain class folders')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to latest checkpoint (default: none)')

# Additional arguments (optional)
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training (default: 0)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use (default: 0)')
parser.add_argument('--arch',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default/supported: resnet18)')
parser.add_argument('--checkpoint_epochs', default=50, type=int,
                    help='Epochs before checkpoint (default: 50)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--num_samples', type=int, default=100000,
                    help='Number of samples to draw from dataset per epoch (default: 100000)')
parser.add_argument('--sh', default=0.2, type=float,
                    help='Simple to hard height for step function update (default: 0.2)')
parser.add_argument('--sw', default=0.25, type=float,
                    help='Simple to hard width for step function update (default: 0.25)')
parser.add_argument('--lr', default=0.03, type=float,
                    help='initial learning rate (default: 0.03)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=100, type=int,
                    help='print frequency as as a function of batchsize/numsamples (default: 100)')
parser.add_argument('--use_hema', action='store_true', default=False,
                    help='Remove use of additional constrain on hematoxylin channel')
parser.add_argument('--lambda_hema', default=10., type=float,
                    help='lambda factor in when adding to loss, only with use_hema (default: 10.)')
# Logging
parser.add_argument('--exp_name', default='exp', type=str,
                    help='Name of the experiment (default: exp)')

# MoCoV2 related-specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')
parser.add_argument('--cos', action='store_true',
                    default=True,
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    cudnn.benchmark = True

    print("******** Define augmentation ********")
    augmentation_base = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    augmentation_adv = transforms.Compose([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("******** Loading datasets ********")
    # Load source dataset
    _, src_dataset, _ = load_dataset(
        transforms_train=sra.loader.TwoCropsTransform(augmentation_base, augmentation_adv, return_hema=args.use_hema),
        transforms_test=None,
        **{'data': args.src_path, 'dataset': args.src_name, 'seed': args.seed}
    )

    # Load target dataset
    _, tar_dataset, _ = load_dataset(
        transforms_train=sra.loader.TwoCropsTransform(augmentation_base, augmentation_adv, return_hema=args.use_hema),
        transforms_test=None,
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
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.use_hema)

    print('Use GPU {}'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    print("******** Define criterion and optimizer ********")
    run_folder = os.path.join("runs", "{}_{}".format(date.today(), args.exp_name))
    filename = "checkpoint_{}_sra".format(args.src_name, args.exp_name)
    criterion_ss = nn.CrossEntropyLoss(reduction='sum').cuda(args.gpu)
    criterion_hema = nn.L1Loss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    writer = SummaryWriter(log_dir=run_folder)
    print_e2h(args)

    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch, args)
        l, l_ind, l_crd, l_hema, top10, top11, top50, top51 = train(train_loader, model, criterion_ss,
                                                                    criterion_hema, optimizer, epoch, args)

        writer.add_scalar('Loss', l, epoch)
        writer.add_scalar('LossIND', l_ind, epoch)
        writer.add_scalar('LossCRD', l_crd, epoch)
        writer.add_scalar('LossHema', l_hema, epoch)
        writer.add_scalars('Top1', {'train_d0': top10, 'train_d1': top11}, epoch)
        writer.add_scalars('Top5', {'train_d0': top50, 'train_d1': top51}, epoch)
        writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], epoch)

        if (epoch+1) % args.checkpoint_epochs == 0:
            print('Saving model epoch: {}'.format(epoch))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args.__dict__,
            }, f='{}_{:04d}.pth.tar'.format(filename, epoch))


def train(train_loader, model, criterion_ss, criterion_hema, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.3e')
    losses_ind = AverageMeter('LossIND', ':.3e')
    losses_crd = AverageMeter('LossCRD', ':.3e')
    losses_hema = AverageMeter('LossHema', ':.3e')
    top1_d0 = AverageMeter('AccD0@1', ':6.2f')
    top1_d1 = AverageMeter('AccD1@1', ':6.2f')
    top5_d0 = AverageMeter('AccD0@5', ':6.2f')
    top5_d1 = AverageMeter('AccD1@5', ':6.2f')

    if args.use_hema:
        l_list = [losses, losses_ind, losses_crd, losses_hema, top1_d0, top1_d1]
    else:
        l_list = [losses, losses_ind, losses_crd, top1_d0, top1_d1],

    progress = ProgressMeter(
        len(train_loader),
        l_list,
        prefix="Epoch: [{}]".format(epoch))

    # Compute simple to hard ratio
    s2h_topk_r = np.floor(epoch/(args.sw*args.epochs))*args.sh
    print('Simple-to-hard consider top: {:.1f}%'.format(s2h_topk_r*100))

    for i, (images, d_set) in enumerate(train_loader):
        # images[0]: key, image[1]: query, image[2]: key_hema, image[3]: query_hema
        # d_set: label source (0) or target (1)

        # Check if batch is composed of tissue from different sources
        if d_set.sum() == 0 or d_set.sum() == args.batch_size:
            continue

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            d_set = d_set.cuda(args.gpu, non_blocking=True)
            if args.use_hema:
                # Only consider q transform
                images[2] = images[2].cuda(args.gpu, non_blocking=True)

        l_ind_d0, t_ind_d0, l_ind_d1, t_ind_d1, h_crd_d0tod1, h_crd_d1tod0, l_hema = model(im_q=images[0],
                                                                                           im_k=images[1],
                                                                                           d_set=d_set)

        # 1. In-domain Self-supervision
        loss_ind_d0 = criterion_ss(l_ind_d0, t_ind_d0)
        loss_ind_d1 = criterion_ss(l_ind_d1, t_ind_d1)
        loss_ind = (1/args.batch_size)*(loss_ind_d0 + loss_ind_d1)

        # 2. Cross-domain self-supervision
        if s2h_topk_r != 0:
            h_crd_d0tod1 = h_crd_d0tod1[h_crd_d0tod1.argsort()[:int(s2h_topk_r*len(h_crd_d0tod1))]]
            h_crd_d1tod0 = h_crd_d1tod0[h_crd_d1tod0.argsort()[:int(s2h_topk_r*len(h_crd_d1tod0))]]
            loss_crd_d0tod1 = h_crd_d0tod1.sum(dim=0)
            loss_crd_d1tod0 = h_crd_d1tod0.sum(dim=0)
            loss_crd = (1 / (args.batch_size*s2h_topk_r)) * (loss_crd_d0tod1 + loss_crd_d1tod0)
        else:
            loss_crd = torch.zeros(1).cuda(args.gpu, non_blocking=True)

        # 3. Overall loss
        loss = loss_ind + loss_crd

        # 4. Reconstruction hema
        if args.use_hema:
            loss_hema = args.lambda_hema * criterion_hema(l_hema, images[2])
            loss = loss + loss_hema

        # 4. Update metrics
        acc1d0, acc5d0 = accuracy(l_ind_d0, t_ind_d0, topk=(1, 5))
        acc1d1, acc5d1 = accuracy(l_ind_d1, t_ind_d1, topk=(1, 5))
        losses_ind.update(loss_ind.item(), images[0].size(0))
        losses_crd.update(loss_crd.item(), images[0].size(0))
        losses_hema.update(loss_hema.item(), images[0].size(0))
        losses.update(loss.item(), images[0].size(0))
        top1_d0.update(acc1d0[0], images[0].size(0))
        top1_d1.update(acc1d1[0], images[0].size(0))
        top5_d0.update(acc5d0[0], images[0].size(0))
        top5_d1.update(acc5d1[0], images[0].size(0))

        # 5. Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, losses_ind.avg, losses_crd.avg, losses_hema.avg, top1_d0.avg, top1_d1.avg, top5_d0.avg, top5_d1.avg


def print_e2h(args):
    es = np.arange(args.epochs)
    rs = np.floor(es / (args.sw * args.epochs)) * args.sh
    idx_steps = [np.nonzero(rs == u)[0][0] for u in np.unique(rs)]
    print("Simple to hard (S2E) stages (epoch: reverse top-k)")
    print("\t->", " | ".join(["e{}: {:.2f}".format(es[idx], rs[idx]) for idx in idx_steps]))


if __name__ == '__main__':
    main()
