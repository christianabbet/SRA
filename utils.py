import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
import torch


def plot_embedding(embedding, cls, cls_labels, filename, vrange=None):
    """
    Plot t-SNE embedding based on input data.
    :param embedding: (N, 2) embedding array.
    :param cls: (N) labels
    :param cls_labels: (D) class labels (names) where len(D) = len(np.unique(cls))
    :param filename: output filename
    :param vrange: [xmin, xmax, ymin, ymax] range for x/y axis in t-SNE plot
    :return: None
    """
    n_labels = np.unique(cls).size

    # Different colormap based on input data size
    if np.unique(cls).__len__() == 2:
        cmap = ListedColormap([
            [1.0, 0.6, 0.333],  # Orange (src)
            [0.267, 0.667, 0.0],  # Green (target)
        ], N=2)
        alpha = 0.8
    else:
        cmap ='Set1'
        alpha = 0.8

    plt.figure(figsize=(8, 6))
    id_rnd = np.random.permutation(len(cls))
    plt.scatter(embedding[id_rnd, 0], embedding[id_rnd, 1], s=7, c=cls[id_rnd], alpha=alpha, cmap=cmap)
    plt.axes().set_aspect('equal')
    cax = plt.colorbar(boundaries=np.arange(max(2, n_labels) + 1) - 0.5)
    cax.set_ticks(np.arange(max(2, n_labels) + 1))
    cax.set_ticklabels(cls_labels)
    if vrange is not None and isinstance(vrange, np.ndarray) and len(vrange) == 4:
        plt.xlim([vrange[0], vrange[2]])
        plt.ylim([vrange[1], vrange[3]])
    plt.title(os.path.basename(filename[:-4]))
    plt.axis('on')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def update_samples(dataset, new_label):
    # Attribute dataset label with new_label value (replace previous label is existing)
    try:
        # If dataset do not have subset dataset
        samples = dataset.samples
        samples = [(s[0], new_label) for s in samples]
        dataset.samples = samples
    except:
        # If dataset do have subset dataset
        samples = dataset.dataset.samples
        samples = [(s[0], new_label) for s in samples]
        dataset.dataset.samples = samples
    return dataset


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'