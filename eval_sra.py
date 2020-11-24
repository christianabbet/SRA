#!/usr/bin/env python
import argparse
import os

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
import numpy as np
from tqdm import tqdm
from utils import plot_embedding
from sklearn.manifold import TSNE
from dataset import load_dataset


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
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-a', '--arch', metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# MoCoV2 related-specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')


def main():
    args = parser.parse_args()

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ]

    print("******** Loading datasets ********")
    # Load source dataset
    src_class_to_idx, src_trainval_dataset, _ = load_dataset(
        transforms_train=transforms.Compose(augmentation),
        transforms_test=None,
        **{'data': args.src_path, 'dataset': args.src_name, 'seed': args.seed}
    )
    src_train_loader = torch.utils.data.DataLoader(
        src_trainval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # Load target dataset
    tar_class_to_idx, tar_trainval_dataset, _ = load_dataset(
        transforms_train=transforms.Compose(augmentation),
        transforms_test=None,
        **{'data': args.tar_path, 'dataset': args.tar_name, 'seed': args.seed}
    )
    tar_train_loader = torch.utils.data.DataLoader(
        tar_trainval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # Build model
    print("******** Building model ********")
    print("Creating model with backbone '{}'".format(args.arch))
    model = sra.builder.SRA(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    print('Use GPU {}'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # Resume from a checkpoint
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        if args.gpu is None:
            checkpoint = torch.load(args.checkpoint)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.checkpoint, map_location=loc)
        state = model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint {} '{}' (epoch {})"
              .format(state, args.checkpoint, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        exit()

    # Compute embedding space if needed
    filename_embedding = os.path.join("sra_eval_{}_to_{}.npy".format(args.src_name, args.tar_name))
    if not os.path.exists(filename_embedding):
        print("Inference ...")
        dsrc_feat, dsrc_lab = eval(src_train_loader, model, len(src_trainval_dataset), args)
        dtar_feat, dtar_lab = eval(tar_train_loader, model, len(tar_trainval_dataset), args)

        print("Save embedding ...")
        data = {
            "checkpoint": args.checkpoint,
            "src_class_to_idx": src_class_to_idx,
            "tar_class_to_idx": tar_class_to_idx,
            "dsrc_feat": dsrc_feat, "dsrc_lab": dsrc_lab,
            "dtar_feat": dtar_feat, "dtar_lab": dtar_lab,
        }
        np.save(filename_embedding, data)

    # Reload data (for consistency and to save time)
    print("Load embedding {} ...".format(filename_embedding))
    n_tsne = 10000
    data = np.load(filename_embedding, allow_pickle=True).item()
    q_feat, q_lab = data['dsrc_feat'], data['dsrc_lab']
    k_feat, k_lab = data['dtar_feat'], data['dtar_lab']

    # Fit t-SNE with both source and target data
    print("Fit t-SNE ...")
    n_feat_subset = np.min([n_tsne, q_feat.shape[0], k_feat.shape[0]])
    data['id_data_d0'] = np.random.RandomState(seed=args.seed).permutation(q_feat.shape[0])[:n_feat_subset]
    data['id_data_d1'] = np.random.RandomState(seed=args.seed).permutation(k_feat.shape[0])[:n_feat_subset]
    data['label_dataset'] = np.concatenate((np.zeros(len(data['id_data_d0'])),
                                            np.ones(len(data['id_data_d1']))), axis=0)

    data['embed'] = TSNE(n_components=2).fit_transform(
        np.concatenate((q_feat[data['id_data_d0']],
                        k_feat[data['id_data_d1']]), axis=0))

    vrange = np.concatenate((data['embed'] .min(axis=0), data['embed'].max(axis=0)), axis=0)
    n_feat_subset = len(data['id_data_d0'])

    # Distribution train and test (dataset label)
    plot_embedding(embedding=data['embed'],
                   cls=data['label_dataset'],
                   cls_labels=[args.src_name, args.tar_name],
                   filename="tsne_{}+{}.png".format(args.src_name, args.tar_name),
                   vrange=vrange)

    # Distribution dataset source
    plot_embedding(embedding=data['embed'][:n_feat_subset],
                   cls=q_lab[data['id_data_d0']],
                   cls_labels=list(data['src_class_to_idx'].keys()),
                   filename="tsne_{}_cls.png".format(args.src_name),
                   vrange=vrange)

    # Distribution dataset target
    plot_embedding(embedding=data['embed'][n_feat_subset:],
                   cls=k_lab[data['id_data_d1']],
                   cls_labels=list(data['tar_class_to_idx'].keys()),
                   filename="tsne_{}_cls.png".format(args.tar_name),
                   vrange=vrange)


def eval(eval_loader, model, n, args):

    # Switch to eval mode and select encoder output
    model_encoder = model.encoder_q
    model_encoder.eval()
    labels = np.zeros(n)
    outputs = np.zeros((n, args.moco_dim))

    # Iterate over dataset
    for i, (images, label) in tqdm(enumerate(eval_loader), total=len(eval_loader)):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        # compute embedding output
        output = model_encoder(images)
        output = nn.functional.normalize(output, dim=1)

        labels[args.batch_size*i:args.batch_size*i+output.shape[0]] = label
        outputs[args.batch_size*i:args.batch_size*i+output.shape[0]] = output.detach().cpu().numpy().squeeze()

    return outputs, labels


if __name__ == '__main__':
    main()
