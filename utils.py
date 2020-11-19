import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


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
