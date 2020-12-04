import numpy as np
from PIL import Image
import torchvision.transforms as transforms


_rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]])
_hed_from_rgb = np.linalg.inv(_rgb_from_hed)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, advanced_transform, return_hema=False):
        self.base_transform = base_transform
        self.advanced_transform = advanced_transform
        self.return_hema = return_hema

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        if self.return_hema:
            q_hema = transforms.ToTensor()(rgb2hed(q)[:, :, 0])
            k_hema = transforms.ToTensor()(rgb2hed(k)[:, :, 0])
            return [self.advanced_transform(q), self.advanced_transform(k), q_hema, k_hema]
        else:
            return [self.advanced_transform(q), self.advanced_transform(k)]


def convert_od2rgb_base10(I):
    """
    Convert from optical density (OD_RGB) to RGB.

    RGB = 255 * log10(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    I = np.maximum(I, 1e-6)
    return (255 * 10**(-1 * I)).astype(np.uint8)


def convert_rgb2od_base10(I):
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * 10^(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = (I == 0)
    I[mask] = 1
    return np.maximum(-1 * np.log10(I / 255), 1e-6)


def rgb2hed(rgb):
    """
    RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.

    :param rgb: (..., 3) ndarray.
        The image in RGB format. Final dimension denotes channels.
    :return:
    out : (..., 3) ndarray
        The image in HED format. Same dimensions as input.

    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
    """
    if isinstance(rgb, Image.Image):
        rgb = np.array(rgb)

    # Convert to optical density
    rgb_od = convert_rgb2od_base10(rgb)

    # Convert to optical density
    hed_od = rgb_od.reshape(-1, 3).dot(_hed_from_rgb)
    hed_od = hed_od.reshape(rgb.shape)

    # Convert back from optical density
    hed = 255-convert_od2rgb_base10(hed_od)
    return hed