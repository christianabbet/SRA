from dataset.wsi import WholeSlideDataset, WholeSlideError
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
import numpy as np
import argparse
from typing import Optional
import yaml
import openslide
from model.sra import SRACls, ResNetCls
from model.utils import get_logger, plot_classification, build_disrete_cmap, save_annotation_qupath
from scipy.special import softmax
from matplotlib import cm
from glob import glob


def load_wsi(wsi_path: str) -> WholeSlideDataset:

    # Build WSIs dataset and data loader
    try:

        # Define transformation. Same for all slides
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.742, 0.616, 0.731], std=[0.211, 0.274, 0.202]),
        ])

        wsi = WholeSlideDataset(
            path=wsi_path,
            transform=transform,
            **config['wsi']
        )

    except WholeSlideError as e:
        # Skip loading image
        return None

    return wsi


def main(
        wsi_path: str,
        model_path: str,
        exp_name: str,
        config: dict,
        use_cuda: Optional[bool] = True,
) -> None:
    """
    Predict classification on WSIs matching input query.

    Parameters
    ----------
    wsi_path: str
        Path to whole slide image.
    model_path: str
        Path to pretrained model (with classification layer).
    config: dict
        Loaded yaml config file.
    use_cuda: bool, optional
        If True, use cuda. Otherwise, rely on CPU.
    """

    logger = get_logger('infer_wsi_classification_sra.log')

    # Use GPU is available
    device = 'cuda' if use_cuda else 'cpu'

    # Load model
    logger.debug('Build and load model from: {}'.format(model_path))
    model = SRACls(**config['model']['parameters'])
    # model = ResNetCls(**config['model']['parameters'])
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()

    # Check if classification exists, otherwise compute it
    if os.path.exists(wsi_path):
        wsis_path = [wsi_path]
    else:
        wsis_path = glob(wsi_path)

    # Iterate over slides
    for p in wsis_path:

        try:

            # put suffix as arg
            logger.debug('Predict output for {}'.format(p))
            output_dir = os.path.join(os.path.dirname(p), "output", exp_name)
            numpy_path = os.path.join(output_dir, os.path.basename(p) + '_{}.npy'.format(exp_name))
            img_path = os.path.join(output_dir, os.path.basename(p) + '_{}.png'.format(exp_name))

            # Create output folder if existing
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            wsi = load_wsi(p)
            logger.debug("Run classification on image ...")
            loader = DataLoader(dataset=wsi, batch_size=config['model']['batch_size'], num_workers=4,
                                shuffle=False, pin_memory=True)

            # Compute classification
            classification = []
            metadata = []

            for crops, metas in tqdm(loader):

                # Only consider first magnification with meta data
                crops = crops[0]
                [mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar] = metas[0]

                # Send to cuda is available
                if use_cuda:
                    crops = crops.cuda()

                # Infer class probabilities
                y_pred = model(crops)

                # Extend results
                classification.extend(y_pred.detach().cpu().numpy())
                metadata.extend(
                    np.array([mag.numpy(), level.numpy(), tx.numpy(), ty.numpy(), cx.numpy(), cy.numpy(), bx.numpy(),
                              by.numpy(), s_src.numpy(), s_tar.numpy()]).T
                )

            # Save results
            data = {
                'name': os.path.basename(wsi_path),
                'wsi_path': wsi_path,
                'model_path': model_path,
                'dataset_name': config['dataset']['name'],
                'classification_labels': config['dataset']['cls_labels'],
                'classification': np.array(classification),
                'metadata_labels': ['mag', 'level', 'tx', 'ty', 'cx', 'cy', 'bx', 'by', 's_src', 's_tar'],
                'metadata': np.array(metadata),
            }
            np.save(file=numpy_path, arr=data)

            # Reload data and plot results
            data = np.load(numpy_path, allow_pickle=True).item()

            if args.plot:
                # Check if classification and output image exist
                logger.debug("Plot result of classification ...")
                plot_classification(
                    image=wsi.s.associated_images['thumbnail'],
                    coords_x=data['metadata'][:, 4],
                    coords_y=data['metadata'][:, 5],
                    cls=np.argmax(data['classification'], axis=1),
                    cls_labels=data['classification_labels'],
                    wsi_dim=wsi.level_dimensions[0],
                    save_path=img_path,
                    cmap=data.get('dataset_name', config['dataset']['name']),  # For old version of *.npy files
                )

            if args.qupath:
                logger.debug("Generate detections for QuPath viz ...")
                # Correction from metadata offset
                offset_x = int(wsi.s.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0))
                offset_y = int(wsi.s.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0))
                # Correction for overlapping tiles
                centering = 0.5 * config['wsi']['padding_factor'] * data['metadata'][0, -2]
                dataset_name = data.get('dataset_name', config['dataset']['name'])

                # Write classification output overlay for QuPath
                save_annotation_qupath(
                    tx=data['metadata'][:, 2] - offset_x + centering,
                    ty=data['metadata'][:, 3] - offset_y + centering,
                    bx=data['metadata'][:, 6] - offset_x - centering,
                    by=data['metadata'][:, 7] - offset_y - centering,
                    values=np.argmax(data['classification'], axis=1),
                    values_name={k: data['classification_labels'][k] for k in range(len(data['classification_labels']))},
                    outpath=os.path.join(img_path[:-4] + "_detection.json"),
                    cmap=build_disrete_cmap(dataset_name),
                )

                for i, cls in enumerate(tqdm(data['classification_labels'], desc='Classes predictions ...')):
                    values = np.clip(softmax(data['classification'], axis=1)[:, i], a_min=0, a_max=0.99)
                    bins = np.linspace(0, 1, 101)
                    save_annotation_qupath(
                        tx=data['metadata'][:, 2] - offset_x + centering,
                        ty=data['metadata'][:, 3] - offset_y + centering,
                        bx=data['metadata'][:, 6] - offset_x - centering,
                        by=data['metadata'][:, 7] - offset_y - centering,
                        values=bins[np.digitize(values, bins)],
                        values_name=bins[np.digitize(values, bins)],
                        outpath=os.path.join(img_path[:-4] + "_{}_detection.json".format(cls)),
                        cmap=cm.get_cmap('inferno'),
                    )

        except Exception as e:
            print('Error handling file: {}'.format(p))
        if not os.path.exists(p):
            raise FileNotFoundError


if __name__ == '__main__':
    """
    Predict WSIs classification using model trained on a Dataset for all slides that match the query. The user can chose 
    to force computation of output if it already exists.

    Examples
    --------
    # Compute images classification based on configuration file
    >>> python infer_wsi_classification_sra.py --wsi_path /path/to/wsi.mrxs --config /path/to/config.cfg
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_path', type=str,
                        default='TCGA-CK-6747-01Z-00-DX1.7824596c-84db-4bee-b149-cd8f617c285f.svs',
                        help='Path to the WSI file (.mrxs, .svs).')
    parser.add_argument('--model_path', type=str,
                        default='best_model_srma_cls_k19.pth',
                        help='Path to the WSI file (.mrxs, .svs).')
    parser.add_argument('--config', type=str,
                        default='conf_wsi_classification_k19.yaml',
                        help='Path to the config yaml file.')
    parser.add_argument('--exp_name', type=str,
                        default='classification_srma',
                        help='Name of the experiment.')
    parser.add_argument('-plot',
                        action='store_true',
                        help='Add argument to plot results as png')
    parser.add_argument('-qupath',
                        action='store_true',
                        help='Add argument to create qupath detection as JSON')

    args = parser.parse_args()

    # Load config file
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    main(
        wsi_path=args.wsi_path,
        model_path=args.model_path,
        exp_name=args.exp_name,
        config=config,
        use_cuda=torch.cuda.is_available()
    )
