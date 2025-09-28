import argparse
import os
import nibabel
import numpy as np

from pathlib import Path
from scipy.ndimage import affine_transform
from tqdm import tqdm

def get_translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    translation_matrix = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return translation_matrix

def get_rotation_matrix(theta: float) -> np.ndarray:

    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    HEART_CLS = 2

    transformation = {
        'T1': get_translation_matrix(275, 200, 0),
        'R1': get_rotation_matrix(-27/180 * np.pi),
        'T3': get_translation_matrix(-275, -200, 0),
        'T4': get_translation_matrix(50, 40, 15),
    }

    order = ['T4','T3','R1','T1']
    initial_transform = np.eye(4)
    for transform in order:
        initial_transform = transformation[transform] @ initial_transform
    inverse_transform = np.linalg.inv(initial_transform)

    dirs = [f for f in os.listdir(src_path) if not f.startswith('.')]
    for patient in tqdm(dirs):
        patient_path = src_path / patient / 'GT.nii.gz'
        gt = nibabel.load(patient_path)
        
        gt_img = gt.get_fdata().astype(gt.get_data_dtype())

        gt_heart = gt_img.copy()
        gt_heart[gt_img != HEART_CLS] = 0
        gt_img[gt_img == HEART_CLS] = 0

        gt_heart = affine_transform(gt_heart, inverse_transform, order=0)

        gt_img[(gt_heart == HEART_CLS) & (gt_img == 0)] = HEART_CLS

        nibabel.save(
            nibabel.Nifti1Image(gt_img,
                                header=gt.header,
                                affine=gt.affine,
                                dtype=gt.get_data_dtype()),
            src_path / patient / 'GT_fixed.nii.gz'
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Transformation parameters")

    parser.add_argument('--source_dir', type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())