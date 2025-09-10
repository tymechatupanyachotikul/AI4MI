import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel
import skimage

from PIL import Image
from utils import map_, tqdm_


"""
TODO: Implement image normalisation.
CT images have a wide range of intensity values (Hounsfield units)
Goal: normalize an image array to the range [0, 255]  and return it as a dtype=uint8
Which is compatible with standard image formats (PNG)
"""
def norm_arr(img: np.ndarray) -> np.ndarray:

    return (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True

def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    assert set(np.unique(gt)) == set(range(5))

    return True


"""
TODO: Implement patient slicing.
Context:
  - Given an ID and paths, load the NIfTI CT volume and (if not test_mode) the GT volume.
  - Validate with sanity_ct / sanity_gt.
  - Normalise CT with norm_arr().
  - Slice the 3D volumes into 2D slices, resize to `shape`, and save PNGs.
  - Currently we have groundtruth masks marked as {0,1,2,3,4} but those values are hard to distinguish in a grayscale png.
    Multiplying by 63 maps them to {0,63,126,189,252}, which keeps labels visually distinct in a grayscale PNG.
    You can use the following code, which works for already sliced 2d images:
    gt_slice *= 63
    assert gt_slice.dtype == np.uint8, gt_slice.dtype
    assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)
  - Return the original voxel spacings (dx, dy, dz).

Hints:
  - Use nibabel to load NIfTI images.
  - Use skimage.transform.resize (tip: anti_aliasing might be useful)
  - The PNG files should be stored in the dest_path, organised into separate subfolders: train/img, train/gt, val/img, and val/gt
  - Use consistent filenames: e.g. f"{id_}_{idz:04d}.png" inside subfolders "img" and "gt"; where idz is the slice index.
"""

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int], test_mode=False) -> tuple[float, float, float]:

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = (id_path / f"{id_}.nii.gz")
    assert id_path.exists()
    assert ct_path.exists()

    # --------- FILL FROM HERE -----------
    ct = nibabel.load(ct_path)
    x,y,z = ct.shape
    dx, dy, dz = ct.header.get_zooms()
    ct_img = ct.get_fdata().astype(ct.get_data_dtype())
    ct_norm = norm_arr(ct_img)

    sanity_ct(ct_img, x, y, z, dx, dy, dz)

    if not test_mode:
        gt = nibabel.load(id_path / "GT.nii.gz")
        gt_img = gt.get_fdata().astype(gt.get_data_dtype())

        sanity_gt(gt_img, ct_img)

    _, _, n = ct_img.shape

    img_dir = dest_path / "img"
    gt_dir = dest_path / "gt"

    if not img_dir.exists():
        img_dir.mkdir(parents=True, exist_ok=True)
    if not gt_dir.exists():
        gt_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        ct_slice = skimage.transform.resize(ct_norm[:, :, i], shape, anti_aliasing=True, preserve_range=True).astype(ct_norm.dtype)
        ct_img_slice = Image.fromarray(ct_slice)
        ct_img_slice.save(dest_path / "img" / f"{id_}_{i:04d}.png")

        if not test_mode:
            gt_slice = skimage.transform.resize(gt_img[:, :, i], shape, order=0, anti_aliasing=False, preserve_range=True).astype(ct_norm.dtype) * 63

            assert gt_slice.dtype == np.uint8, gt_slice.dtype
            assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

            gt_img_slice = Image.fromarray(gt_slice)
            gt_img_slice.save(dest_path / "gt" / f"{id_}_{i:04d}.png")

    return dx, dy, dz

"""
TODO: Implement a simple train/val split.
Requirements:
  - List patient IDs from <src_path>/train (folder names).
  - Shuffle them (respect a seed set in main()).
  - Take the first `retains` as validation, and the rest as training.
  - Return (training_ids, validation_ids).
"""

def get_splits(src_path: Path, retains: int) -> tuple[list[str], list[str]]:
    # TODO: your code here
    patient_ids = [p.name for p in (src_path / "train").iterdir() if p.is_dir()]
    random.shuffle(patient_ids)
    validation_ids = patient_ids[:retains]
    training_ids = patient_ids[retains:]

    return training_ids, validation_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)
    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    assert src_path.exists()
    assert dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    training_ids, validation_ids = get_splits(src_path, args.retains)


    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape))

        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        resolutions = list(map(pfun, iterator))

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())