import logging
from typing import Tuple, Optional, List, Union
import PIL
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Colormap
from shapely.geometry import Polygon
import json
import numpy as np


def get_logger(logfile='log.log'):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

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


def plot_classification(
        image: PIL.Image,
        coords_x: np.ndarray,
        coords_y: np.ndarray,
        cls: np.ndarray,
        cls_labels: np.ndarray,
        wsi_dim: Tuple[int, int],
        save_path: str,
        cmap: Optional[str] = 'kather19',
) -> None:
    """
    Create a plot compose of 2 subplot representing the original image and the classification result.

    Parameters
    ----------
    image: PIL.Image
        Thumbnail of the whole slide image.
    coords_x: np.ndarray of shape (n_samples, )
        x coordinates of the patches in the wsi referential.
    coords_y: np.ndarray of shape (n_samples, )
        y coordinates of the patches in the wsi referential.
    cls: np.ndarray of shape (n_samples, )
        Classes of each classified patch.
    cls_labels: np.ndarray of shape (n_classes, )
        Name of the classes.
    wsi_dim: tuple of int
        Dimension of the original image slide. Should correspond to coord referential.
    save_path: str
        Output path to image to save. Can be JPEG, PNG, PDF.
    cmap: str, optional
        Name of the colomap to use for classification display. Default is `k19`
    """

    # Generate map
    map = build_prediction_map(coords_x, coords_y, cls[:, None], wsi_dim=wsi_dim)[:, :, 0]

    # Create plot dimensions and scaling factor for scatter plot
    fig_size = (int(2 * 12 * image.size[0] / image.size[1]), 12)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=fig_size, gridspec_kw={'width_ratios': [1, 1, 0.05]})
    plt.suptitle(os.path.basename(save_path))

    # Plot reference image
    axes[0].set_title('Image thumbnail')
    axes[0].axis('off')
    axes[0].imshow(image)
    # Plot classification map
    axes[1].set_title('Image classification')
    axes[1].axis('off')
    r = axes[1].imshow(
        map + 1,  # Plot map with offset of 1 (avoid background = -1)
        cmap=build_disrete_cmap(cmap, background=[[1.0, 1.0, 1.0]]),  # choose the background color for cls = 0
        interpolation='nearest',  # Avoid interpolation aliasing when scaling image
        vmin=0, vmax=len(cls_labels),
    )

    # Define color bar with background color
    cls_new_labels = np.concatenate((['-'], cls_labels))
    cax = fig.colorbar(
        r,  # reference map for color
        cax=axes[2],  # axis to use to plot image
        orientation='vertical',  # orientation of the colorbar
        boundaries=np.arange(len(cls_new_labels) + 1) - 0.5  # Span range for color
    )
    cax.set_ticks(np.arange(len(cls_new_labels)))  # Define ticks position (center of colors)
    cax.set_ticklabels(cls_new_labels)  # Define names of labels

    # Save plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# Build map
def build_prediction_map(
        coords_x: np.ndarray,
        coords_y:  np.ndarray,
        feature:  np.ndarray,
        wsi_dim: Optional[tuple] = None,
        default: Optional[float] = -1.,
) -> np.ndarray:
    """
    Build a prediction map based on x, y coordinate and feature vector. Default values if feature is non existing
    for a certain location is -1.

    Parameters
    ----------
    coords_x: np.ndarray of shape (N,)
        Coordinates of x points.
    coords_y: np.ndarray of shape (N,)
        Coordinates of y points.
    feature: np.ndarray of shape (N, M)
        Feature vector.
    wsi_dim: tuple of int, optional
        Size of the original whole slide. The function add a margin around the map if not null. Default value is None.
    default: float, optional
        Value of the pixel when the feature is not defined.

    Returns
    -------
    map: np.ndarray (W, H, M)
        Feature map. The unaffected points use the default value -1.
    """
    # Compute offset of coordinates in pixel (patch intervals)
    interval_x = np.min(np.unique(coords_x)[1:] - np.unique(coords_x)[:-1])
    interval_y = np.min(np.unique(coords_y)[1:] - np.unique(coords_y)[:-1])

    # Define new coordinates
    if wsi_dim is None:
        offset_x = np.min(coords_x)
        offset_y = np.min(coords_y)
    else:
        offset_x = np.min(coords_x) % interval_x
        offset_y = np.min(coords_y) % interval_y

    coords_x_ = ((coords_x - offset_x) / interval_x).astype(int)
    coords_y_ = ((coords_y - offset_y) / interval_y).astype(int)

    # Define size of the feature map
    if wsi_dim is None:
        map = default * np.ones((coords_y_.max() + 1, coords_x_.max() + 1, feature.shape[1]))
    else:
        map = default * np.ones((int(wsi_dim[1] / interval_y), int(wsi_dim[0] / interval_x), feature.shape[1]))

    # Affect values to map
    map[coords_y_, coords_x_] = feature

    return map


def build_disrete_cmap(name: str, background: Optional[np.ndarray] = None) -> Colormap:
    """
    Build colormap for displaying purpose. Could be one of : 'k19'

    Parameters
    ----------
    name: str
        Name of the colormap to build. Should be one of : 'kather19', 'crctp'
    background: np.ndarray, optional
        Color of th background. THis color will be added as the first item of the colormap. Default value is None.

    Returns
    -------
    cmap: Colormap
        The corresponding colormap
    """

    if name == 'kather19':
        colors = np.array([
            [247, 129, 191],  # Pink - Adipose
            [153, 153, 153],  # Gray - Back
            [255, 255, 51],  # Yellow - Debris
            [152, 78, 160],  # Purple - Lymphocytes
            [255, 127, 0],  # Orange - Mucus
            [23, 190, 192],  # Cyan - Muscle
            [166, 86, 40],  # Brown - Normal mucosa
            [55, 126, 184],  # Blue - Stroma
            [228, 26, 28],  # Red - Tumor
        ]) / 255
    elif name == 'kather19crctp':
        colors = np.array([
            [247, 129, 191],  # Pink - Adipose
            [153, 153, 153],  # Gray - Back
            [255, 255, 51],  # Yellow - Debris
            [152, 78, 160],  # Purple - Lymphocytes
            [255, 127, 0],  # Orange - Mucus
            [23, 190, 192],  # Cyan - Muscle
            [166, 86, 40],  # Brown - Normal mucosa
            [55, 126, 184],  # Blue - Stroma
            [228, 26, 28],  # Red - Tumor
            [77, 167, 77],  # Green - Complex Stroma
        ]) / 255
    elif name == 'embedding':
        colors = np.array([
            [1.0, 0.6, 0.333],  # Orange (src)
            [0.267, 0.667, 0.0],  # Green (target)
        ])
    else:
        # Set of 8 colors (see https://matplotlib.org/stable/tutorials/colors/colormaps.html)
        colors = np.array(cm.get_cmap('Accent').colors)

    if background is not None:
        colors = np.concatenate((background, colors), axis=0)

    cmap = ListedColormap(colors, name='cmap_k19', N=colors.shape[0])
    return cmap


def build_poly(tx: np.ndarray, ty: np.ndarray, bx: np.ndarray, by: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Counter clock-wise
    px = np.vstack((tx, bx, bx, tx)).T
    py = np.vstack((ty, ty, by, by)).T

    return px, py


def save_annotation_qupath(
        tx: np.ndarray,
        ty: np.ndarray,
        bx: np.ndarray,
        by: np.ndarray,
        values: np.ndarray,
        values_name: Union[np.ndarray, dict],
        outpath: str,
        cmap: Colormap,
) -> None:
    """

    Parameters
    ----------
    tx: array_like
    ty: array_like
    bx: array_like
    by: array_like
    values: array_like
    values_name: array_like
    outpath: str
    cmap: Colormap

    """

    # Check dimensions
    if not all(tx.shape == np.array([ty.shape, bx.shape, by.shape, values.shape])):
        return

    # Build shape and simplify the shapes if True
    polys_x, polys_y = build_poly(tx=tx, ty=ty, bx=bx, by=by)

    # Extract outer shapes
    coords = {}
    colors = []
    clss = []
    for i in range(len(polys_x)):
        color = 255*np.array(cmap(values[i]))[:3]
        colors.append(color)
        label = values[i] if isinstance(values_name, np.ndarray) else values_name[values[i]]
        clss.append([label])
        coords['poly{}'.format(i)] = {
            "coords": np.vstack((polys_x[i], polys_y[i])).tolist(),
            "class": str(label),
            "color": [int(color[0]), int(color[1]), int(color[2])]
        }

    with open(outpath, 'w') as outfile:
        json.dump(coords, outfile)
