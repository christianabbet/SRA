#!/usr/bin/env python
import argparse
import os
import matplotlib.pyplot as plt

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
from sklearn.manifold import TSNE
from dataset import load_dataset


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
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


parser.add_argument('-a', '--arch', metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed',
                    # action='store_true',
                    action='store_true',
                    default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--cos', action='store_true',
                    default=True,
                    help='use cosine lr schedule')


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

    # Build model and load model weights
    print("******** Building model ********")
    print("Creating model with backbone '{}'".format(args.arch))
    model = sra.builder.SRA(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    print(model)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    folder_output = os.path.join(os.path.dirname(args.resume), "embed")
    filename_embedding = os.path.join(folder_output, "moco_eval_{}_to_{}.npy".format(
        args.src_name, args.tar_name))

    if not os.path.exists(filename_embedding):
        d0_feat, d0_lab = eval(src_train_loader, model, len(src_trainval_dataset), args, output_mlp=use_mlp, use_sup=use_sup)
        d1_feat, d1_lab = eval(tar_train_loader, model, len(tar_trainval_dataset), args, output_mlp=use_mlp, use_sup=use_sup)

        if not os.path.exists(folder_output):
            os.makedirs(folder_output)

        data = {
            "checkpoint": args.resume,
            "class_to_idx": src_class_to_idx,
            "class_to_idx2": tar_class_to_idx,
            "use_mlp": use_mlp, "use_cds": use_cds,
            "d0_feat": d0_feat, "d0_lab": d0_lab,
            "d1_feat": d1_feat, "d1_lab": d1_lab,
        }

        np.save(filename_embedding, data)

    # ################### RUN Multiple experiments #########################
    data = np.load(filename_embedding, allow_pickle=True).item()
    n_tsne = 10000

    # ################### RUN Query and retrieval #########################
    augmentation_viz = [
        transforms.Resize(224),
    ]
    reset_transfrom(src_trainval_dataset, augmentation_viz)
    reset_transfrom(tar_trainval_dataset, augmentation_viz)

    q_feat, q_lab = data['d0_feat'], data['d0_lab']
    k_feat, k_lab = data['d1_feat'], data['d1_lab']

    plot_query(folder_output, src_trainval_dataset, tar_trainval_dataset, q_feat, q_lab, k_feat, k_lab, src_class_to_idx,
               tar_class_to_idx, args.dataset, args.dataset2, n_queries_per_cls=5, n_queries_top=8, seed=0)

    plot_query(folder_output, tar_trainval_dataset, src_trainval_dataset, k_feat, k_lab, q_feat, q_lab, tar_class_to_idx,
               src_class_to_idx, args.dataset2, args.dataset, n_queries_per_cls=5, n_queries_top=8, seed=0)

    # ################## RUN TSNE ##################

    if 'embed' not in data:
        # Make the two sets the same size
        n_feat_subset = np.min([n_tsne, q_feat.shape[0], k_feat.shape[0]])
        data['id_data_d0'] = np.random.permutation(q_feat.shape[0])[:n_feat_subset]
        data['id_data_d1'] = np.random.permutation(k_feat.shape[0])[:n_feat_subset]

        # 1. Fit No normalization
        # Fit TSNE with both
        data['label_dataset'] = np.concatenate((np.zeros(len(data['id_data_d0'])),
                                                np.ones(len(data['id_data_d1']))), axis=0)

        data['embed'] = TSNE(n_components=2).fit_transform(
            np.concatenate((q_feat[data['id_data_d0']],
                            k_feat[data['id_data_d1']]), axis=0))

        np.save(filename_embedding, data)

    vrange = np.concatenate((data['embed'] .min(axis=0), data['embed'].max(axis=0)), axis=0)
    n_feat_subset = len(data['id_data_d0'])

    # CLS_KATHER16 = [(0, 'tumor', '01_TUMOR'), (1, 'stroma', '02_STROMA'), (2, 'complex', '03_COMPLEX'),
    #                 (3, 'lympho', '04_LYMPHO'), (4, 'debris', '05_DEBRIS'), (5, 'mucosa', '06_MUCOSA'),
    #                 (6, 'adipose', '07_ADIPOSE'), (7, 'empty', '08_EMPTY')]
    #
    # CLS_KATHER19 = [(0, 'adipose', 'ADI'), (1, 'background', 'BACK'), (2, 'debris', 'DEB'),
    #                 (3, 'lymphocytes', 'LYM'), (4, 'mucus', 'MUC'), (5, 'muscle', 'MUS'),
    #                 (6, 'normal mucosa', 'NORM'), (7, 'cancer stroma', 'STR'), (8, 'tumor', 'TUM')]
    #
    # x = 6.4
    # y = -43.87
    # coord = np.array([x, y])
    # top_items_k16 = np.linalg.norm(data['embed'][:n_feat_subset] - coord, axis=1).argsort()
    # top_items_k19 = np.linalg.norm(data['embed'][n_feat_subset:] - coord, axis=1).argsort()
    #
    # n = 10
    # fig, axes = plt.subplots(2, n, figsize=(2 * n, 2 * 2))
    # for i in range(axes.shape[1]):
    #     id_k16 = data['id_data_d0'][top_items_k16][i]
    #     id_k19 = data['id_data_d1'][top_items_k19][i]
    #     axes[0, i].imshow(trainval_dataset[id_k16][0])
    #     axes[0, i].set_title(trainval_dataset[id_k16][1])
    #     axes[1, i].imshow(trainval_dataset2[id_k19][0])
    #     axes[1, i].set_title(trainval_dataset2[id_k19][1])

    # Distribution train and test (dataset label)
    plot_embedding(embedding=data['embed'],
                   cls=data['label_dataset'],
                   cls_labels=[args.dataset, args.dataset2],
                   filename=os.path.join(folder_output, "tsne_{}+{}_mlp{}.png".format(
                       args.dataset, args.dataset2, int(use_mlp))),
                   vrange=vrange)

    # Distribution train and test (cls label)
    plot_embedding(embedding=data['embed'],
                   cls=np.concatenate([
                       data['d0_lab'][data['id_data_d0']], data['d1_lab'][data['id_data_d1']]]),
                   cls_labels=list(data['class_to_idx'].keys()),
                   filename=os.path.join(folder_output, "tsne_{}+{}_cls_mlp{}.png".format(
                       args.dataset, args.dataset2, int(use_mlp))),
                   vrange=vrange)

    # Distribution dataset1
    plot_embedding(embedding=data['embed'][:n_feat_subset],
                   cls=q_lab[data['id_data_d0']],
                   cls_labels=list(data['class_to_idx'].keys()),
                   filename=os.path.join(folder_output, "tsne_{}_cls_mlp{}.png".format(
                       args.dataset, int(use_mlp))),
                   vrange=vrange)

    # Distribution dataset2
    plot_embedding(embedding=data['embed'][n_feat_subset:],
                   cls=k_lab[data['id_data_d1']],
                   cls_labels=list(data['class_to_idx2'].keys()),
                   filename=os.path.join(folder_output, "tsne_{}_cls_mlp{}.png".format(
                       args.dataset2, int(use_mlp))),
                   vrange=vrange)

    if (args.dataset == 'kather16' and args.dataset2 == 'kather19') \
            or (args.dataset == 'kather19' and args.dataset2 == 'kather16'):
        plot_cls_k1916(data, data['id_data_d0'], data['id_data_d1'], data['embed'], n_feat_subset, src_class_to_idx,
                       filename=os.path.join(folder_output, "tsne_{}_k16-19_mlp{}.png".format(
                           args.dataset2, int(use_mlp))))


def plot_cls_k1916(data, id_data_d0, id_data_d1, embed, n_feat_subset, class_to_idx, filename):

    # ############################# Overall
    class_to_marker2 = dict({0: 6, 1: 7, 2: 4, 3: 3, 4: 4, 5: 1, 6: 5, 7: 1, 8: 0})
    label_datasets = np.concatenate((np.zeros(len(id_data_d0)), np.ones(len(id_data_d1))), axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5),
                             gridspec_kw={"width_ratios": [1, 1, 0.05, 1, 0.05]})
    # fig.subplots_adjust(wspace=0.5)

    # SOURCE Domain
    z = embed[n_feat_subset:]
    c = [class_to_marker2[int(l)] for l in data['d1_lab'][id_data_d1]]
    c0 = axes[0].scatter(z[:, 0], z[:, 1], c=np.array(c), cmap="Set1", s=2, alpha=1, vmax=7, vmin=0, zorder=2)
    # sns.kdeplot(embed[n_feat_subset:], embed[n_feat_subset:],
    #             cmap="Greens", shade=True, shade_lowest=False, alpha=0.5, ax=axes[0])
    # axes[0].axis('off')

    # TARGET Domain
    # sns.kdeplot(embed[:n_feat_subset], embed[:n_feat_subset],
    #             cmap="Oranges", shade=True, shade_lowest=False, alpha=0.5, ax=axes[1])
    z = embed[:n_feat_subset]
    c = data['d0_lab'][id_data_d0]
    c1 = axes[1].scatter(z[:, 0], z[:, 1], c=np.array(c), cmap="Set1", s=2, alpha=1, vmax=7, vmin=0, zorder=2)
    # axes[1].axis('off')

    # Both Domain
    z = embed
    c = label_datasets
    c2 = axes[3].scatter(z[:, 0], z[:, 1], c=c, cmap="bwr", s=2, alpha=1, vmax=1, vmin=0)

    cax = fig.colorbar(c0, cax=axes[2])
    cax.set_ticks(np.arange(max(2, 8) + 1))
    cax.set_ticklabels(list(class_to_idx.keys()))

    cax = fig.colorbar(c2, cax=axes[4])
    cax.set_ticks([0, 1])
    cax.set_ticklabels(['Kather16', 'Kather19'])

    # axes[1].axis('off')
    plt.suptitle(os.path.basename(filename[:-4]))
    plt.savefig(filename)
    plt.tight_layout()
    plt.close()

    # ############################# Overall

    plt.figure(figsize=(12, 10))
    z = embed
    c = np.concatenate(
        (data['d0_lab'][id_data_d0],
         [class_to_marker2[int(l)] for l in data['d1_lab'][id_data_d1]]), axis=0)
    id_rnd = np.random.permutation(len(c))
    plt.scatter(z[id_rnd, 0], z[id_rnd, 1], c=np.array(c)[id_rnd], cmap="Set1", s=15, alpha=1, vmax=7, vmin=0, zorder=2)
    plt.axis('off')
    plt.axes().set_aspect('equal')
    plt.savefig(filename+"_empty.jpeg", bbox_inches='tight')
    plt.close()


def eval(eval_loader, model, n, args, output_mlp=False, use_sup=False):

    # switch to eval mode
    if not use_sup:
        if output_mlp:
            model_encoder = model.encoder_q
            out_dim = 128
        else:
            model_encoder = nn.Sequential(*list(model.encoder_q.children())[:-1])
            out_dim = 512
    else:
        model_encoder = model
        model_encoder.fc = nn.Sequential()
        out_dim = 512

    model_encoder.eval()
    labels = np.zeros(n)
    outputs = np.zeros((n, out_dim))

    for i, (images, label) in tqdm(enumerate(eval_loader), total=len(eval_loader)):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model_encoder(images)
        if output_mlp:
            output = nn.functional.normalize(output, dim=1)

        labels[args.batch_size*i:args.batch_size*i+output.shape[0]] = label
        outputs[args.batch_size*i:args.batch_size*i+output.shape[0]] = output.detach().cpu().numpy().squeeze()

    return outputs, labels




if __name__ == '__main__':
    main()
