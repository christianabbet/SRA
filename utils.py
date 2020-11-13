import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_embedding(embedding, cls, cls_labels, filename, vrange=None):
    n_labels = np.unique(cls).size

    if np.unique(cls).__len__() == 2:
        cmap = ListedColormap([
            [1.0, 0.6, 0.333],  # Orange (src)
            [0.267, 0.667, 0.0],  # Green (target)
        ], N=2)
        alpha = 0.9
    else:
        cmap ='Set1'
        alpha = 0.9

    # First generate empty version for latex
    plt.figure(figsize=(12, 10))
    id_rnd = np.random.permutation(len(cls))
    plt.scatter(embedding[id_rnd, 0], embedding[id_rnd, 1], s=10, c=cls[id_rnd], alpha=alpha, cmap=cmap)
    plt.axis('off')
    plt.axes().set_aspect('equal')
    plt.savefig(filename+"_empty.jpeg", bbox_inches='tight')

    # Then add content for debugging
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