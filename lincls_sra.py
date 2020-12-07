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
from utils import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy
from sklearn.metrics import f1_score, accuracy_score

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Main arguments (mandatory)
parser = argparse.ArgumentParser(description='SRA evaluation process')
parser.add_argument('--src_name', type=str, default="kather19",
                    choices=["kather19"],
                    help='Name of the source dataset')
parser.add_argument('--src_path', type=str, default="",
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
parser.add_argument('--ratio_label', default=0.01, type=float,
                    help='Label ratio to use (default: 0.01)')
parser.add_argument('--ratio_val', default=0.2, type=float,
                    help='Label ratio split between train and validation (default: 0.2)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--weight_decay', default=0., type=float,
                    help='weight decay (default: 0.)')
parser.add_argument('--lr', default=1, type=float,
                    help='initial learning rate (default: 1.)')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--print-freq', default=100, type=int,
                    help='print frequency as as a function of batchsize/numsamples (default: 100)')

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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    args = parser.parse_args()
    cudnn.benchmark = True

    print("******** Define augmentation ********")
    augmentation_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    augmentation_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("******** Loading datasets ********")
    # Load source dataset
    class_to_idx, trainval_dataset, test_dataset = load_dataset(
        transforms_train=augmentation_train,
        transforms_test=augmentation_test,
        **{'data': args.src_path, 'dataset': args.src_name, 'seed': args.seed}
    )

    print("******** Fraction data train: {} ********".format(args.ratio_label))
    rnd = np.random.RandomState(seed=args.seed)
    rnd_subset = rnd.permutation(len(trainval_dataset))[:int(args.ratio_label*len(trainval_dataset))]
    rnd_subset_train = rnd_subset[int(len(rnd_subset)*args.ratio_val):]
    rnd_subset_val = rnd_subset[:int(len(rnd_subset)*args.ratio_val)]
    train_dataset = torch.utils.data.Subset(trainval_dataset, rnd_subset_train)
    val_dataset = torch.utils.data.Subset(trainval_dataset, rnd_subset_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Build model
    print("******** Building model ********")
    print("Creating model with backbone '{}'".format(args.arch))
    model = sra.builder.SRA(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    print('Use GPU {}'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    if not os.path.exists(args.checkpoint):
        raise Exception("Path to checkpoint {} does not exists".format(args.checkpoint))
    print("Loading and assigning weights ...")
    state_dict = torch.load(args.checkpoint)['state_dict']
    model.load_state_dict(state_dict, strict=True)

    for name, param in model.named_parameters():
        param.requires_grad = False

    print("Freeze weights and add linear classifier  ...")
    out_dim = model.encoder_q_mlp[-1].out_features
    n_cls = len(class_to_idx)

    module_cls = []
    module_cls.extend(list(model.encoder_q_mlp))
    module_cls.append(nn.Linear(in_features=out_dim, out_features=n_cls).cuda())
    module_cls[-1].weight.data.normal_(mean=0.0, std=0.01)
    module_cls[-1].bias.data.zero_()

    model.encoder_q_mlp = nn.Sequential(*module_cls)

    print("******** Define criterion and optimizer ********")
    run_folder = os.path.join("runs", "{}_cls_{}".format(date.today(), args.exp_name))
    filename = "checkpoint_{}_sra_cls".format(args.src_name, args.exp_name)
    writer = SummaryWriter(log_dir=run_folder)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print("Number of parameters to optimize: {}".format(len(parameters)))
    optimizer = torch.optim.SGD(parameters, args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        losses_tr, top1_tr, top5_tr, y_target_tr, y_output_tr = train(train_loader, model, 'train', criterion,
                                                                      optimizer, epoch, args)
        losses_va, top1_va, top5_va, y_target_va, y_output_va = train(val_loader, model, 'eval', criterion,
                                                                      optimizer, epoch, args)
        losses_te, top1_te, top5_te, y_target_te, y_output_te = train(test_loader, model, 'eval', criterion,
                                                                      optimizer, epoch, args)

        writer.add_scalars('Loss', {'train': losses_tr, 'val': losses_va, 'test': losses_te}, epoch)
        writer.add_scalars('Top1', {'train': top1_tr, 'val': top1_va, 'test': top1_te}, epoch)
        writer.add_scalars('Top1', {'train': top1_tr, 'val': top1_va, 'test': top1_te}, epoch)
        writer.add_scalars('F1', {'train': f1_score(y_target_tr, np.argmax(y_output_tr, axis=1), average='weighted'),
                                  'val': f1_score(y_target_va, np.argmax(y_output_va, axis=1), average='weighted'),
                                  'test': f1_score(y_target_te, np.argmax(y_output_te, axis=1), average='weighted')},
                           epoch)


def train(train_loader, model, mode, criterion, optimizer, epoch, args):

    losses = AverageMeter('Loss', ':.3e')
    top1 = AverageMeter('Acc@1', ':.3e')
    top5 = AverageMeter('Acc@5', ':.3e')
    y_target = []
    y_output = []
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    for i, (images, target) in enumerate(train_loader):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if mode == 'train':
            optimizer.zero_grad()

        # compute output
        qh = model.encoder_q(images)  # queries: NxC
        qh = qh.view((-1, model.dim_encoder, 7, 7))
        q = nn.functional.adaptive_avg_pool2d(qh, (1, 1)).squeeze()
        output = model.encoder_q_mlp(q)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if mode == 'train':
            loss.backward()
            optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)

        y_target.extend(target.detach().cpu().numpy())
        y_output.extend(output.detach().cpu().numpy())

    return losses.avg, top1.avg, top5.avg, np.array(y_target), np.array(y_output)


if __name__ == '__main__':
    main()
