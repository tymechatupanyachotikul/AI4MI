#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path
import nibabel
import numpy as np
from PIL import Image
import skimage
from tqdm import tqdm

def main(args: argparse.Namespace):

    src_dir = Path(args.data_folder)
    dest_dir = Path(args.dest_folder)
    grp_regex = re.compile(args.grp_regex)
    source_scan_pattern = args.source_scan_pattern
    scale = 255 // (args.num_classes - 1)
    if not dest_dir.exists():
        os.mkdir(dest_dir)

    assert src_dir.exists()

    out = {}
    for patient_slice in tqdm(os.listdir(src_dir)):
        match = re.match(grp_regex, patient_slice)

        if match:
            patient_id = match.group(1)

            if patient_id not in out:
                gt_path = source_scan_pattern.format(id_=patient_id)
                gt = nibabel.load(gt_path)
                x, y, z = gt.shape

                out[patient_id] = {
                    'pred': None,
                    'gt': None
                }
                out[patient_id]['pred'] = np.zeros((x, y, z), dtype=gt.get_data_dtype())
                out[patient_id]['gt'] = gt

            slice = np.array(Image.open(src_dir / Path(patient_slice)), dtype=out[patient_id]['pred'].dtype) / scale
            slice = skimage.transform.resize(slice, out[patient_id]['pred'].shape[:2], order=0,anti_aliasing=False, preserve_range=True).astype(
                out[patient_id]['pred'].dtype)
            out[patient_id]['pred'][:, :, int(os.path.splitext(patient_slice)[0][-4:])] = slice

    for patient, slice in out.items():
        nibabel.save(
            nibabel.Nifti1Image(slice['pred'],
                                header=slice['gt'].header,
                                affine=slice['gt'].affine,
                                dtype=slice['gt'].get_data_dtype()),
            dest_dir / f'{patient}.nii.gz'
        )

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stitch 2D slices into 3D volumes')
    parser.add_argument('--data_folder', type=Path, required=True,
                        help="Name of the data folder with sliced data.")
    parser.add_argument('--dest_folder', type=Path,
                        help="Name of the destination folder with stitched data")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes.")
    parser.add_argument("--grp_regex", type=str,
                        help="Pattern for the filename.")
    parser.add_argument("--source_scan_pattern", type=str,
                        help="Pattern to the original scans to get original size.")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
