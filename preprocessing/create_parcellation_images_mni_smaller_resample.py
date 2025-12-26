"""
This module creates parcelation 4D images from ENGIMA PTSD
using the FC time-series data. use here template flow for MNI mask, always the same for each subject

Create subject-level ROI-parcellated 4D fMRI-like volumes in MNI space from TSV time series.

This module:
1) Resamples a group atlas (e.g., Schaefer2011Combined, Brainnetome) to a target mask grid.
2) Intersects atlas labels with a (template) brain mask to remove out-of-brain voxels.
3) Optionally dilates ROIs within the mask and removes tiny ROI fragments.
4) Converts atlas TSV time series (T × N_ROI) into a 4D NIfTI (X × Y × Z × T),
  where each voxel inside an ROI is filled with that ROI’s time series.

Intended for ENIGMA-PTSD resting-state derivatives organized by site/subject.
"""

import os
import sys
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger
from nilearn.image import resample_img

# for now test with this two and continue later # DONT DO THIS USE THE SEG
# MASK YOU ALREADY GIVE FROM DUKE PEOPLE!
from nilearn.image import resample_to_img
from scipy.ndimage import binary_dilation
from templateflow import api

# To ignore all warnings:
warnings.filterwarnings("ignore")
# To ignore specific categories of warnings (e.g., DeprecationWarning):
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_decimated_grid(reference_img: nib.Nifti1Image, factor: int):
    """
    Build a new (lower-res) grid from reference_img by increasing voxel sizes
    by `factor` and shrinking the shape accordingly.

    Returns:
      target_affine, target_shape
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")

    ref_aff = reference_img.affine.copy()
    ref_shape = np.array(reference_img.shape[:3], dtype=int)

    # Voxel sizes from affine (absolute scale)
    voxel_sizes = np.sqrt((ref_aff[:3, :3] ** 2).sum(axis=0))
    new_voxel_sizes = voxel_sizes * factor

    # Keep orientation, only scale the zooms
    R = ref_aff[:3, :3].copy()
    for i in range(3):
        R[:, i] = R[:, i] / voxel_sizes[i] * new_voxel_sizes[i]

    target_affine = ref_aff.copy()
    target_affine[:3, :3] = R

    # Shrink shape (ceil to cover FOV)
    target_shape = tuple(np.ceil(ref_shape / factor).astype(int))

    return target_affine, target_shape

def load_timeseries_tsv(path, n_rois):
    """
    Load an atlas time-series TSV file (no header) as a numeric array.

    The TSV is expected to be shaped as (T, N_columns). If extra columns exist,
    only the first `n_rois` are retained (useful when files include additional
    nuisance regressors or concatenated outputs).

    Args:
      path: Path to a tab-separated file containing time series (no header).
      n_rois: Number of ROI columns expected/required.

    Returns:
      A nparray of shape (T, n_rois) containing the ROI time series.

    Raises:
      ValueError: If the TSV has fewer than 'n_rois' columns.
    """
    ts = pd.read_csv(path, sep="\t", header=None).values  # (T, N_ROI)
    if ts.shape[1] < n_rois:
        raise ValueError(f"Expected at least {n_rois} columns, got {ts.shape[1]}")
    # if there are extra columns, keep first n_rois for Brainnetome
    return ts[:, :n_rois]


def personalize_atlas_for_subject(
    atlas_path, subj_mask_path, out_path=None, min_voxels=10, dilate=False, dilation_iters=1, decimate_factor= 1
):
    """
    Create a subject-specific (mask-constrained) atlas by resampling and masking.

    Workflow:
    1) Resample the input integer-labeled atlas into the mask image grid
       using nearest-neighbor interpolation (preserves ROI labels).
    2) Zero out voxels outside the subject/target brain mask.
    3) Optionally apply a small ROI-wise dilation constrained to the mask to
       patch small holes without flooding the entire brain.
    4) Remove ROI fragments smaller than `min_voxels`.

    Args:
      atlas_path: Path to an integer-labeled atlas NIfTI in (typically) MNI space.
      subj_mask_path: Path to a brain/fMRI mask NIfTI defining the target grid/space.
      out_path: Output path for the masked/resampled atlas NIfTI.
      min_voxels: Minimum voxel count required to keep an ROI after masking.
      dilate: If True, apply constrained binary dilation per ROI label.
      dilation_iters: Number of dilation iterations (keep small; e.g., 1).

    Returns:
      atlas: 3D integer NumPy array of atlas labels in mask space.
      mask_img: The loaded mask NIfTI image (for affine/header reuse).

    Side Effects:
      Writes the personalized atlas NIfTI to `out_path` if provided
    """
    # 1) Resample Schaefer to subject mask grid
    atlas_img = nib.load(str(atlas_path))
    mask_img = nib.load(str(subj_mask_path))

    # --- NEW: build decimated target mask grid ---
    if decimate_factor > 1:
        tgt_aff, tgt_shape = make_decimated_grid(mask_img, decimate_factor)

        # Resample mask into the low-res grid
        # Option A (keeps binary-ish): nearest
        mask_low = resample_img(
            mask_img,
            target_affine=tgt_aff,
            target_shape=tgt_shape,
            interpolation="nearest",
        )
        # If your mask is probabilistic and you prefer smoother boundaries:
        # mask_low = resample_img(mask_img, tgt_aff, tgt_shape, interpolation="continuous")

        mask_img_use = mask_low
    else:
        mask_img_use = mask_img

    img_resampled = resample_to_img(
        atlas_img,
        mask_img_use,
        interpolation="nearest",  # preserves integer labels
    )
    atlas = img_resampled.get_fdata().astype(int)

    mask_data = mask_img_use.get_fdata()
    # If you used "continuous" interpolation above, threshold:
    mask = mask_data > 0

    # 2) Intersect with subject mask
    atlas[~mask] = 0

    # 3) OPTIONAL: small dilation if you want to close tiny gaps,
    #    but *not* to fill the entire brain.
    if dilate:
        roi_labels = np.unique(atlas)
        roi_labels = roi_labels[roi_labels != 0]

        dilated_atlas = atlas.copy()
        for label in roi_labels:
            roi_mask = atlas == label
            if not roi_mask.any():
                continue
            # Dilate within the brain mask
            roi_dil = binary_dilation(roi_mask, iterations=dilation_iters)
            roi_dil = roi_dil & mask  # stay inside brain
            dilated_atlas[roi_dil] = label

        atlas = dilated_atlas

    # 4) Drop tiny fragments
    roi_labels = np.unique(atlas)
    roi_labels = roi_labels[roi_labels != 0]

    for label in roi_labels:
        n_vox = np.sum(atlas == label)
        if n_vox < min_voxels:
            atlas[atlas == label] = 0

    out_img = nib.Nifti1Image(atlas.astype(np.int16), mask_img.affine, mask_img.header)
    nib.save(out_img, str(out_path))

    logger.info(f"Saved personalized atlas to {out_path}")

    return atlas, mask_img


# THESE atlases are not needed now. # COMMENT THIS FOR NOW**
# get the schaefer atlas 2018 ROIs here
# atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)

# get the power 2011 atlass coords here
# coords = fetch_coords_power_2011()


path_data = "../../Data/RSData/"
mni_mask_path = api.get("MNI152NLin2009cAsym", desc="brain", suffix="mask", extension=".nii.gz")


# do the function here to create a new 4D parcelation image with the
# information of the time_series combined in the 400 ROIs from schaefer
def tsv_to_4d_parcellation(tsv_path, subj_atlas, affine, header, out_path, n_rois=None):
    """
    Convert ROI time series TSV data into a 4D NIfTI volume using an atlas label map.

    Each ROI label in `subj_atlas` is filled with the corresponding ROI time series
    across all timepoints, producing a 4D volume (X, Y, Z, T). Voxels with label 0
    remain zero.

    Label convention:
       ROI labels are assumed to be 1-indexed (label 1 maps to column 0).

    Args:
      tsv_path: Path to a TSV file containing time series (T × N_columns), no header.
      subj_atlas: 3D integer label array (already resampled/masked for the subject).
      affine: Affine transform for the output image (typically from the mask).
      header: Header for the output image (typically from the mask).
      out_path: Output path for the 4D NIfTI (.nii or .nii.gz).
      n_rois: If provided, use only the first `n_rois` columns and assume labels
          are in 1..n_rois. If None, infer `n_rois` from max(subj_atlas).

    Raises:
        ValueError: If the TSV has fewer columns than required by 'n_rois' / atlas labels.
    """
    # Load time series
    ts = pd.read_csv(tsv_path, sep="\t", header=None).values  # shape (T, N_cols)
    T, n_cols = ts.shape

    if n_rois is None:
        n_rois = int(np.max(subj_atlas))
    if n_cols < n_rois:
        raise ValueError(f"TSV has {n_cols} cols but atlas has labels up to {n_rois}")

    # Keep only first n_rois columns (in case of 'Combined' TSV with extra columns)
    ts = ts[:, :n_rois]  # (T, n_rois)

    # Prepare 4D volume (X, Y, Z, T)
    x, y, z = subj_atlas.shape
    vol_4d = np.zeros((x, y, z, T), dtype=np.float32)

    roi_labels = np.unique(subj_atlas)
    roi_labels = roi_labels[roi_labels != 0]

    for label in roi_labels:
        roi_mask = subj_atlas == label
        roi_ts = ts[:, label - 1]  # time series for this ROI (T,)
        roi_ts = np.nan_to_num(roi_ts, nan=0.0)

        # Broadcast: assign ts to all voxels of this ROI over time
        vol_4d[roi_mask, :] = roi_ts

    img_4d = nib.Nifti1Image(vol_4d, affine, header)
    nib.save(img_4d, str(out_path))
    logger.info(f"Saved 4D ROI-time-series parcellation to {out_path}")


"""
  **Main section of the code here**
"""

# define the atlas files here
schaefer_atlas_file = "../../Atlases/tpl-MNI152NLin2009cAsym_atlas-schaefer2011Combined_dseg.nii.gz"
power_atlas_file = "../../Atlases/tpl-MNI152NLin2009cAsym_atlas-power2011_dseg.nii.gz"
brainnetome_atlas_file = (
    "../../Atlases/tpl-MNI152NLin2009cAsym_atlas-brainnetomeCombined_dseg.nii.gz"
)

subjects_missing = [
    "sub-S26",
    "sub-S66",
    "sub-S15",
    "sub-S62",
    "sub-S68",
    "sub-S50",
    "sub-S06",
    "sub-S71",
    "sub-S49",
    "sub-S43",
    "sub-S34",
    "sub-S39",
    "sub-S33",
    "sub-S76",
    "sub-S01",
    "sub-S20",
    "sub-S57",
    "sub-S18",
]

# get the decimation factor here and validates it
if len(sys.argv) >= 2:
    decimation_factor_val = int(sys.argv[1])
    decimation = int(sys.argv[1])
else:
    decimation_factor_val = []
    decimation =  4

# get here first the sites names to continue processing - as first level folders
for site in os.listdir(path_data):
    if "." not in site:
        # do a for loops across the whole subjects
        for subjects in os.listdir(path_data + "/" + site):
            # get the brain mask for each subject
            if site == "Beijing":
                subject_val = subjects
            elif site == "Capetown":
                subject_val = subjects.replace("-capetown", "").replace("-tygerberg", "")
            elif site == "Cisler":
                subject_val = subjects.replace("_", "").replace("-D", "-d").replace("-P", "-p")
            elif site == "Masaryk":
                subject_val = subjects.replace("C", "c")
            elif site == "UMN":
                subject_val = subjects.replace("_", "").replace("M", "m")
            elif site == "Ghent":
                subject_val = subjects.replace("-S", "-s")
            elif site == "Toledo":
                subject_val = subjects.replace("-M", "-m").replace("O", "o")
            elif site == "Tours":
                subject_val = subjects.replace("T", "t").replace("V", "v")
            elif site == "NanjingYixing":
                if "SD" in subjects:
                    subject_val = subjects.replace("S", "s")
            elif site == "Leiden":
                subject_val = subjects.replace("-S", "-s").replace("E", "e")
            elif site == "Michigan":
                subject_val = subjects.replace("M", "m").replace("R", "r")
            else:
                subject_val = subjects

            # VALIDATES IF THE PATH EXIST TO NOT OVERWRITE**
            if "." not in subjects and not os.path.exists(
                path_data
                + "/"
                + site
                + "/"
                + subjects
                + f"/{subject_val}_schaefer_mni_image_small{decimation_factor_val}.nii.gz"
            ):  # validates if the file exist and if not it can do preprocessing!!
                # masks_path + '/' + site + '/' + subject_mask + '-preproc_brain_mask.nii.gz'
                mask_file = str(mni_mask_path[0])

                # check this just for vanderbilt!!
                if not site == "Vanderbilt":
                    # check the preffix here
                    schaefer_file = (
                        subject_val
                        + "_"
                        + "task-rest_feature-corrMatrix1_atlas-schaefer2011Combined_timeseries.tsv"
                    )
                    power_file = (
                        subject_val
                        + "_"
                        + "task-rest_feature-corrMatrix1_atlas-power2011_timeseries.tsv"
                    )
                    brainnetome_file = (
                        subject_val
                        + "_"
                        + "task-rest_feature-corrMatrix1_atlas-brainnetomeCombined_timeseries.tsv"
                    )

                else:
                    # check the preffix here
                    schaefer_file = (
                        subject_val
                        + "_"
                        + "task-rest_feature-corrMatrix1_atlas-schaefer2011CombinedDseg_timeseries.tsv"
                    )
                    power_file = (
                        subject_val
                        + "_"
                        + "task-rest_feature-corrMatrix1_atlas-power2011CombinedDseg_timeseries.tsv"
                    )
                    brainnetome_file = (
                        subject_val
                        + "_"
                        + "task-rest_feature-corrMatrix1_atlas-brainnetomeCombinedDseg_timeseries.tsv"
                    )

                if os.path.exists(mask_file):
                    path_file_schaefer_file = (
                        path_data + "/" + site + "/" + subjects + "/" + schaefer_file
                    )
                    path_file_power_file = (
                        path_data + "/" + site + "/" + subjects + "/" + power_file
                    )
                    path_file_brainnetome_file = (
                        path_data + "/" + site + "/" + subjects + "/" + brainnetome_file
                    )

                    if not os.path.exists(
                        path_data + "/" + site + "/" + subjects + "/" + schaefer_file
                    ) or not os.path.exists(
                        path_data + "/" + site + "/" + subjects + "/" + brainnetome_file
                    ):
                        # check if the path doesnt exist so choose the other denoising
                        # method
                        path_file_schaefer_file = (
                            path_data
                            + "/"
                            + site
                            + "/"
                            + subjects
                            + "/"
                            + schaefer_file.replace("x1", "x")
                        )
                        path_file_power_file = (
                            path_data
                            + "/"
                            + site
                            + "/"
                            + subjects
                            + "/"
                            + power_file.replace("x1", "x")
                        )
                        path_file_brainnetome_file = (
                            path_data
                            + "/"
                            + site
                            + "/"
                            + subjects
                            + "/"
                            + brainnetome_file.replace("x1", "x")
                        )
                        logger.error(
                            "Initial method for denoising doesn't exist choosing the alternative.."
                        )

                    # 1) personalize the evaluation for the schaefer images - leave the
                    # dilation in False and min_voxels in 20
                    subj_atlas_schaefer, mask_subj_img_schaefer = personalize_atlas_for_subject(
                        atlas_path=schaefer_atlas_file,
                        subj_mask_path=mask_file,
                        out_path=path_data
                        + "/"
                        + site
                        + "/"
                        + subjects
                        + f"/{subject_val}_schaefer_mni_image_small{decimation_factor_val}.nii.gz",
                        min_voxels=20,
                        dilate=False,
                        dilation_iters=1,
                        decimate_factor=decimation
                    )
                    tsv_to_4d_parcellation(
                        tsv_path=path_file_schaefer_file,
                        subj_atlas=subj_atlas_schaefer,
                        affine=mask_subj_img_schaefer.affine,
                        header=mask_subj_img_schaefer.header,
                        out_path=path_data
                        + "/"
                        + site
                        + "/"
                        + subjects
                        + f"/{subject_val}_schaefer_4d_mni_image_small{decimation_factor_val}.nii.gz",
                        n_rois=434,
                    )

                    # SKIP power for now!!
                    # subj_atlas_power, mask_subj_img_power = personalize_atlas_for_subject(atlas_path=power_atlas_file, subj_mask_path=mask_file, out_path=path_data + '/' + site + '/' + subjects + f'/{subjects}_power_image.nii.gz', min_voxels=20, dilate=False, dilation_iters=1)
                    # tsv_to_4d_parcellation(tsv_path=path_file_power_file, subj_atlas=subj_atlas_power, affine=mask_subj_img_power.affine, header=mask_subj_img_power.header,  out_path=path_data + '/' + site + '/' + subjects + f'/{subjects}_power_4d_image.nii.gz', n_rois=264)

                    # 2) personalize the evaluation for the brainnetome images - leave
                    # the dilation in False and min_voxels in 20
                    subj_atlas_brainnetome, mask_subj_img_brainnetome = (
                        personalize_atlas_for_subject(
                            atlas_path=brainnetome_atlas_file,
                            subj_mask_path=mask_file,
                            out_path=path_data
                            + "/"
                            + site
                            + "/"
                            + subjects
                            + f"/{subject_val}_brainnetome_mni_image_small{decimation_factor_val}.nii.gz",
                            min_voxels=20,
                            dilate=False,
                            dilation_iters=1,
                            decimate_factor=decimation
                        )
                    )
                    tsv_to_4d_parcellation(
                        tsv_path=path_file_brainnetome_file,
                        subj_atlas=subj_atlas_brainnetome,
                        affine=mask_subj_img_brainnetome.affine,
                        header=mask_subj_img_brainnetome.header,
                        out_path=path_data
                        + "/"
                        + site
                        + "/"
                        + subjects
                        + f"/{subject_val}_brainnetome_4d_mni_image_small{decimation_factor_val}.nii.gz",
                        n_rois=263,
                    )

                    logger.success(f"Images from {subjects} in site {site} have been processed!!")
                else:
                    logger.error(
                        f"Mask for subject {subjects} does not exist in the folder!!.. continue..."
                    )

        logger.success(f"Images from site {site} ALL have been processed!!")
