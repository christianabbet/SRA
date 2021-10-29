from skimage.morphology import remove_small_objects, disk, dilation
from typing import Optional, List, Tuple, Callable
from cv2 import cvtColor, blur, COLOR_RGB2GRAY
from torch.utils.data import Dataset
import numpy as np
import openslide
import PIL
import pathlib


LUT_MAGNIFICATION_X = [20, 40]
LUT_MAGNIFICATION_MPP = [0.5, 0.25]


class WholeSlideError(Exception):
    pass


class WholeSlideDataset(Dataset):

    def __init__(
            self,
            path: str,
            crop_sizes_px: Optional[List[int]] = None,
            crop_magnifications: Optional[List[float]] = None,
            transform: Optional[Callable] = None,
            padding_factor: Optional[float] = 0.5,
            remove_background: Optional[bool] = True,
            remove_oob: Optional[bool] = True,
            remove_alpha: Optional[bool] = True,
            ratio_object_thresh: Optional[float] = 1e-3,
    ) -> None:
        """
        Load a crop as a dataset format. The object is iterable.

        Parameters
        ----------
        path: str
            Path to the whole slide as a "*.tif, *.svs, *.mrxs format"
        crop_sizes_px: list of int, optional
            List of crops output size in pixel, default value is [224].
        crop_magnifications: list of float, optional
            List of crops magnification level, default value is [20].
        transform: callable, optional
            Transformation to apply to crops, default value is None. So far, only one augmentation for all crops
            is possible.
        padding_factor: float, optional
            Padding value when creating reference grid. Distance between two consecutive crops as a proportion of the
            first listed crop size. Default value is 0.5.
        remove_background: bool, optional
            Remove background crops if their average intensity value is below the threshold value (240). Default value
            is True.
        remove_oob: bool, optional
            Remove all crops where its representation at a specific magnification is out of bound (out of the scanned
            image). Default value is True.
        remove_alpha: bool, optional
            Remove alpha channel when extracting patches to create a RGB image (instead of RGBA). More suitable to ML
            input transforms. Default value is True.
        ratio_object_thresh: float, optional
            Size of the object ot remove. THe value isexpressed as a ratio with respect to the area of the whole slide.
            Default value is 1e-3 (e.i., 1%).

        Raises
        ------
        WholeSlideError
            If it is not possible to load the WSIs.

        Examples
        --------
        Load a slide with tiling at different magnifications (20x, 10x, 5x) and sizes (224, 224, 244).
        >>> wsi = WholeSlideDataset(
                path="/path/to/slide.svs/.mrxs",
                crop_sizes_px=[224, 224, 224],
                crop_magnifications=[20., 10., 5.],
            )
        """

        extension = pathlib.Path(path).suffix
        if extension != '.svs' and extension != '.mrxs' and extension != '.tif':
            raise NotImplementedError("Only *.svs, *.tif and *.mrxs files supported")

        # Load and create slide and affect default values
        self.path = path
        self.s = openslide.open_slide(self.path)
        self.crop_sizes_px = crop_sizes_px
        self.crop_magnifications = crop_magnifications
        self.transform = transform
        self.padding_factor = padding_factor
        self.remove_alpha = remove_alpha
        self.mask = None

        if self.crop_sizes_px is None:
            self.crop_sizes_px = [224]

        if self.crop_magnifications is None:
            self.crop_magnifications = [20]

        # Dimension of the slide at different levels
        self.level_dimensions = self.s.level_dimensions
        # Down sampling factor at each level
        self.level_downsamples = self.s.level_downsamples
        # Get average micro meter per pixel (MPP) for the slide
        try:
            self.mpp = 0.5 * (float(self.s.properties[openslide.PROPERTY_NAME_MPP_X]) +
                              float(self.s.properties[openslide.PROPERTY_NAME_MPP_Y]))
        except KeyError:
            raise WholeSlideError('No resolution found in WSI metadata. Impossible to build pyramid.')

        # Extract level magnifications
        self.level_magnifications = self._get_magnifications(self.mpp, self.level_downsamples)
        # Consider reference level as the level with highest resolution
        self.crop_reference_level = 0

        # Build reference grid / crop centers
        self.crop_reference_cxy = self._build_reference_grid(
            crop_size_px=self.crop_sizes_px[0],
            crop_magnification=self.crop_magnifications[0],
            padding_factor=padding_factor,
            level_magnification=self.level_magnifications[self.crop_reference_level],
            level_shape=self.level_dimensions[self.crop_reference_level]
        )

        # Assume the whole slide has an associated image
        if remove_background and 'thumbnail' in self.s.associated_images:
            # Extract image thumbnail from slide metadata
            img_thumb = self.s.associated_images['thumbnail']
            # Get scale factor compared to reference size
            mx = img_thumb.size[0] / self.level_dimensions[self.crop_reference_level][0]
            my = img_thumb.size[1] / self.level_dimensions[self.crop_reference_level][1]
            # Compute foreground mask
            self.mask = self._foreground_mask(img_thumb, ratio_object_thresh=ratio_object_thresh)
            # Select subset of point that are part of the foreground
            id_valid = self.mask[
                np.round(my * self.crop_reference_cxy[:, 1]).astype(int),
                np.round(mx * self.crop_reference_cxy[:, 0]).astype(int),
            ]
            self.crop_reference_cxy = self.crop_reference_cxy[id_valid]

        # Build grid for all levels
        self.crop_metadatas = self._build_crop_metadatas(
                self.crop_sizes_px,
                self.crop_magnifications,
                self.level_magnifications,
                self.crop_reference_cxy,
                self.crop_reference_level,
        )

        # Remove samples that are oob from sampling
        if remove_oob:
            # Compute oob sa,ples
            oob_id = self._oob_id(self.crop_metadatas, self.level_dimensions[self.crop_reference_level])
            # Select only smaples that are within bounds.
            self.crop_reference_cxy = self.crop_reference_cxy[~oob_id]
            self.crop_metadatas = self.crop_metadatas[:, ~oob_id]

    @staticmethod
    def _pil_rgba2rgb(image: PIL.Image, default_background: Optional[List[int]] = None) -> PIL.Image:
        """
        Convert RGBA image to RGB format using default background color.
        From https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil/9459208#9459208

        Parameters
        ----------
        image: PIL.Image
            Input RBA image to convert.
        default_background: list of int, optional
            Value to us as background hen alpha channel is not 255. Default value is white (255, 255, 255).
        Returns
        -------
        Image with alpha channel removed.

        """
        if default_background is None:
            default_background = (255, 255, 255)

        image.load()
        background = PIL.Image.new('RGB', image.size, default_background)
        background.paste(image, mask=image.split()[3])
        return background

    @staticmethod
    def _oob_id(
            crop_grid: np.ndarray,
            level_shape: List[int],
    ) -> np.ndarray:
        """
        Check is the samples are within bounds.

        Parameters
        ----------
        crop_grid: array_like
            Input crop meta data of C element where C is the number of crops. For each crop
        level_shape: list of int
            Dimension of the image
        Returns
        -------

        """
        # Extract top left coordinated
        tx, ty = crop_grid[:, :, 2], crop_grid[:, :, 3]
        # Extract top right coordinated
        bx, by = crop_grid[:, :, 4], crop_grid[:, :, 5]
        # Check for boundaries
        oob_id = (tx < 0) | (ty < 0) | (bx > level_shape[0]) | (by > level_shape[1])
        return np.any(oob_id, axis=0)

    @staticmethod
    def _build_crop_metadatas(
            crop_sizes_px: List[int],
            crop_magnifications: List[float],
            level_magnifications: List[float],
            crop_reference_grid: np.ndarray,
            crop_reference_level: int,
    ) -> np.ndarray:
        """
        Build metadata for each crops definitions.

        Parameters
        ----------
        crop_sizes_px: list of int, optional
            List of crops output size in pixel, default value is [224].
        crop_magnifications: list of float, optional
            List of crops magnification level, default value is [20].
        level_magnifications: list of float
            List of available magnifications (one for each level)
        crop_reference_grid:
            Reference grid with shape [Nx2] where N is the number of samples. The column represent x and y coordinates
            of the center of the crops respectively.
        crop_reference_level: int
            Reference level used to compute the reference grid.

        Returns
        -------
        metas: array_like
            Meta data were each entry correspond to the metadata a the crop and [mag, level, tx, ty, cx, cy, bx,
            by, s_src, s_tar]. With mag = magnification of the crop, level = level at which the crop was extracted,
            (tx, ty) = top left coordinate of the crop, (cx, cy) = center coordinate of the crop, (bx, by) = bottom
            right coordinates of the crop, s_src = size of the crop at the level, s_tar = siz of the crop after
            applying rescaling.
        """

        crop_grids = []
        for t_size, t_mag in zip(crop_sizes_px,  crop_magnifications):
            # Level that we use to extract current slide region
            t_level = WholeSlideDataset._get_optimal_level(t_mag, level_magnifications)
            # Scale factor between the reference magnification and the magnification used
            t_scale = level_magnifications[t_level] / level_magnifications[crop_reference_level]
            # Final image size at the current level / magnification
            t_level_size = t_size / (t_mag / level_magnifications[t_level])
            # Offset to recenter image
            t_shift = (t_level_size / t_scale) // 2
            # Return grid as format: [level, tx, ty, bx, by, level_size, size]
            grid_ = np.concatenate(
                (
                    t_mag * np.ones(len(crop_reference_grid))[:, np.newaxis],  # Magnification
                    t_level * np.ones(len(crop_reference_grid))[:, np.newaxis],  # Level
                    crop_reference_grid - t_shift,  # (tx, ty) coordinates
                    crop_reference_grid,  # (cx, cy) coordinates values
                    crop_reference_grid + t_shift,  # (bx, by) coordinates
                    t_level_size * np.ones(len(crop_reference_grid))[:, np.newaxis],  # original images size
                    t_size * np.ones(len(crop_reference_grid))[:, np.newaxis],  # target image size
                ), axis=1
            )
            crop_grids.append(grid_)
        return np.array(crop_grids)

    @staticmethod
    def _get_optimal_level(
            magnification: float,
            level_magnifications: List[float]
    ) -> int:
        """
        Estimate the optimal level to extract crop. It the wanted level do nt exist, use a level with higher resolution
        (lower level) and resize crop.

        Parameters
        ----------
        magnification: float
            Wanted output magnification
        level_magnifications: list of float
            List of available magnifications (one for each level)
        Returns
        -------
        optimal_level: int
            Estimated optimal level for crop extraction
        """

        # Get the highest level that is a least as high resolution as the wanted target.
        if magnification <= np.max(level_magnifications):
            optimal_level = np.nonzero(np.array(level_magnifications) >= magnification)[0][-1]
        else:
            # If no suitable candidates are found, use max resolution
            optimal_level = 0
            print('Slide magnifications {} do not match expected target magnification {}'.format(
                magnification, level_magnifications))

        return optimal_level

    @staticmethod
    def _get_magnifications(
        mpp: float,
        level_downsamples: List[float],
        error_max: Optional[float] = 1e-1,
    ) -> List[float]:
        """
        Compute estimated magnification for each level. The computation rely on the definition of LUT_MAGNIFICATION_X
        and LUT_MAGNIFICATION_MPP that are mapped. For example the assumption is 20x -> ~0.5MPP and 40x -> ~0.25MPP.

        Parameters
        ----------
        mpp: float
            Resolution of the slide (and the scanner).
        level_downsamples: lost of float
            Down sampling factors for each level as a list of floats.
        error_max: float, optional
            Maximum relative error accepted when trying to match magnification to predefined factors. Default value
            is 1e-1.
        Returns
        -------
        level_magnifications: list of float
            Return the estimated magnifications for each level.
        """

        error_mag = np.abs((np.array(LUT_MAGNIFICATION_MPP) - mpp) / mpp)
        # if np.min(error_mag) > error_max:
        #     print('Error too large for mpp matching: mpp={}, error={}'.format(mpp, np.min(error_mag)))

        return LUT_MAGNIFICATION_X[np.argmin(error_mag)] / np.round(level_downsamples).astype(int)

    @staticmethod
    def _foreground_mask(
            img: PIL.Image.Image,
            intensity_thresh: Optional[int] = 240,
            ratio_object_thresh: Optional[float] = 1e-4
    ) -> np.ndarray:
        """
        Compute foreground mask the slide base on the input image. Usually the embedded thumbnail is used.

        Parameters
        ----------
        img: PIL.Image.Image
            Downscaled version of the slide as a PIL image
        intensity_thresh: int
            Intensity threshold applied on te grayscale version of the image to distinguish background from foreground.
            The default value is 240.
        ratio_object_thresh: float
            Minimal ratio of the object to consider as a relevant region. Ratio is applied on the area of the object.

        Returns
        -------
        mask: np.ndarray
            Masked version of the input image where '0', '1' indicates regions belonging to background and foreground
            respectively.
        """

        # Convert image to grayscale
        mask = cvtColor(np.array(img), COLOR_RGB2GRAY)
        # Blur image to remove hih frequencies
        mask = blur(mask, (5, 5))
        # Apply threshold on background intensity
        mask = mask < intensity_thresh
        # Remove smallest object as a ratio of original image size
        mask = remove_small_objects(mask, min_size=ratio_object_thresh*np.prod(mask.shape))
        # Add final margin to avoid cutting edges
        disk_edge = np.ceil(np.max(mask.shape)*ratio_object_thresh).astype(int)
        mask = dilation(mask, disk(max(1, disk_edge)))

        return mask

    @staticmethod
    def _build_reference_grid(
            crop_size_px: int,
            crop_magnification: float,
            padding_factor: float,
            level_magnification: float,
            level_shape: List[int],
    ) -> np.ndarray:
        """
        Build reference grid for cropping location. The grid is usually computed at the lowest magnification.

        Parameters
        ----------
        crop_size_px: int
            Output size in pixel.
        crop_magnification: float
            Magnification value.
        padding_factor: float
            Padding factor to use. Define the interval between two consecutive crops.
        level_magnification: float
            Selected magnification.
        level_shape: list of int
            Size of the image at the selected level.
        Returns
        -------
        (cx, cy): list of int
            Center coordinate of the crop.
        """

        # Define the size of the crop at the selected level
        level_crop_size_px = int((level_magnification/crop_magnification) * crop_size_px)

        # Compute the number of crops for each dimensions (rows and columns)
        n_w = np.floor((1/padding_factor) * (level_shape[0]/level_crop_size_px - 1)).astype(int)
        n_h = np.floor((1/padding_factor) * (level_shape[1]/level_crop_size_px - 1)).astype(int)

        # Compute the residual margin at each side of the image
        margin_w = int(level_shape[0] - padding_factor*(n_w-1)*level_crop_size_px) // 2
        margin_h = int(level_shape[1] - padding_factor*(n_h-1)*level_crop_size_px) // 2

        # Compute the final center for the cropping
        c_x = (np.arange(n_w) * level_crop_size_px * padding_factor + margin_w).astype(int)
        c_y = (np.arange(n_h) * level_crop_size_px * padding_factor + margin_h).astype(int)
        c_x, c_y = np.meshgrid(c_x, c_y)

        return np.array([c_x.flatten(), c_y.flatten()]).T

    def __len__(self) -> int:
        return len(self.crop_reference_cxy)

    def __getitem__(self, idx: int) -> Tuple[List[object], List[object]]:
        """
        Get slide element as a function of the index idx.

        Parameters
        ----------
        idx: int
            Index of the crop

        Returns
        -------
        imgs: List of PIL.Image
            List of extracted crops for this index.
        metas: List of List of float
            Meta data were each entry correspond to the metadata a the crop and [mag, level, tx, ty, cx, cy, bx,
            by, s_src, s_tar]. With mag = magnification of the crop, level = level at which the crop was extracted,
            (tx, ty) = top left coordinate of the crop, (cx, cy) = center coordinate of the crop, (bx, by) = bottom
            right coordinates of the crop, s_src = size of the crop at the level, s_tar = siz of the crop after
            applying rescaling.
        """
        imgs = []
        metas = []
        for i in range(len(self.crop_sizes_px)):
            # Extract metadata for crops
            mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar = self.crop_metadatas[i][idx]
            # Extract crop
            img = self.s.read_region((int(tx), int(ty)), int(level), size=(int(s_src), int(s_src)))
            # If needed, resize crop to match output shape
            if s_src != s_tar:
                img = img.resize((int(s_tar), int(s_tar)))
            # Append images and metadatas
            if self.remove_alpha:
                img = self._pil_rgba2rgb(img)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            metas.append([mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar])

        return imgs, metas
