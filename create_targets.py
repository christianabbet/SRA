from dataset.wsi import WholeSlideDataset
from torch.utils.data import Subset
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import os


def main(data_query: str, export: str, n_subset: int, size_limit: int):
    wsi_paths = glob(data_query)

    for n, p in enumerate(wsi_paths):

        print("[{}/{}]: {}".format(n+1, len(wsi_paths), p))
        try:

            # If file os too small
            if os.path.getsize(p)//1e6 < size_limit:
                print('Skip slide too small')
                continue

            output_subfolder = os.path.join(export, os.path.basename(p))
            os.makedirs(output_subfolder, exist_ok=True)

            # Create WSI
            wsi = WholeSlideDataset(
                path=p,
                crop_sizes_px=[224],
                crop_magnifications=[20],
                padding_factor=1,  # No overlap
                transform=None,
            )

            # Randomly sample n_subset patches from WSI
            rnd = np.random.RandomState(seed=0)
            idx = rnd.permutation(len(wsi))[:min(len(wsi), n_subset)]

            # Extract images
            for i, (img, meta) in enumerate(tqdm(Subset(wsi, idx), desc="Extract sub-image")):
                patch_name = "{}_{}_x{}_y{}.jpeg".format(os.path.basename(p), idx[i], int(meta[0][4]), int(meta[0][5]))
                img[0].save(os.path.join(output_subfolder, patch_name))

        except Exception as e:
            print('*** Unexpected error as: {} ***'.format(e))


if __name__ == '__main__':
    """

    Examples
    --------
    # Export WSIs to tiles 
    # >>> python create_wsi_dataset.py --data_query="/path/to/wsis/.mrxs" --export=/path/to/export" --n_subset=1000
    """

    parser = argparse.ArgumentParser(description='Run CNN classifier on Kather 19')
    parser.add_argument('--data_query', dest='data_query', type=str,
                        default='/home/abbet/Desktop/Bern*/*.mrxs',
                        help='Path to main data folder containing slides in sub-folders')
    parser.add_argument('--export', dest='export', type=str,
                        # default='data/GENERATED_TARGETS',
                        help='Export newly created set to folder')
    parser.add_argument('--n_subset', dest='n_subset', type=int, default=np.inf,
                        help='Number of patches to consider per slide')
    parser.add_argument('--size_limit', dest='size_limit', type=int, default=0,
                        help='Size limit for WSIs (in MB). Could be useful to avoid low res images.')
    args = parser.parse_args()

    # Load config file

    main(args.data_query, args.export, args.n_subset, args.size_limit)
