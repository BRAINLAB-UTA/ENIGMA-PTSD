"""
This module creates parcelation 4D images from ENGIMA PTSD
using the FC time-series data.
"""

import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger

# for now test with this two and continue later # DONT DO THIS USE THE SEG
# MAKS YOU ALREADY GIVE FROM DUKE PEOPLE!
from nilearn.image import resample_to_img

# from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation

# To ignore all warnings:
warnings.filterwarnings("ignore")
# To ignore specific categories of warnings (e.g., DeprecationWarning):
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_timeseries_tsv(path, n_rois):
    """
    Load a time series TSV with NO header.
    Returns an array of shape (T, N_ROI).
    """
    ts = pd.read_csv(path, sep="\t", header=None).values  # (T, N_ROI)
    if ts.shape[1] < n_rois:
        raise ValueError(f"Expected at least {n_rois} columns, got {ts.shape[1]}")
    # if there are extra columns, keep first n_rois for Brainnetome
    return ts[:, :n_rois]


def personalize_atlas_for_subject(
    atlas_path, subj_mask_path, out_path=None, min_voxels=10, dilate=False, dilation_iters=1
):
    """
    Intersects a group atlas with a subject-specific mask to create a
    'personalized' atlas for that subject.

    atlas_path: NIfTI with integer labels (e.g., Schaefer, Brainnetome, HO)
    subj_mask_path: NIfTI mask (brain / fMRI mask) in the same space
    out_path: optional output NIfTI for subject-specific atlas
    min_voxels: ROIs with fewer voxels than this after masking are zeroed out

    Returns:
        subject_mask (3D np.array, int)
        atlas_path (nibabel Nifti1Image) – for affine/header reuse
    """
    """
    Map Schaefer atlas into subject mask space with high fidelity:

    1) Resample Schaefer to subject mask grid (same shape+affine).
    2) Intersect with subject brain mask.
    3) Optionally do a SMALL dilation, but only inside the mask.
       (No full "fill the whole mask" extrapolation.)

    schaefer_path: Schaefer NIfTI in MNI (labels 1..400)
    subj_mask_path: subject brain/fMRI mask (NIfTI) in target space
    out_atlas_path: output NIfTI for subject-space Schaefer
    min_voxels: drop ROIs with fewer voxels than this
    dilate: if True, do a small dilation to patch tiny holes
    dilation_iters: number of dilation iterations (keep this small, e.g. 1)
    """
    # 1) Resample Schaefer to subject mask grid
    atlas_img = nib.load(str(atlas_path))
    mask_img = nib.load(str(subj_mask_path))

    img_resampled = resample_to_img(
        atlas_img,
        mask_img,
        interpolation="nearest",  # preserves integer labels
    )
    atlas = img_resampled.get_fdata().astype(int)
    mask = mask_img.get_fdata() > 0

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


# THESE atlases are not needed now..
# get the schaefer atlas 2018 ROIs here
# atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)

# get the power 2011 atlass coords here
# coords = fetch_coords_power_2011()


path_data = "/media/jmm/One_Touch/UTA_Lab/ENIGMA-PTSD/RSdata/"
masks_path = "/media/jmm/One_Touch/UTA_Lab/ENIGMA-PTSD/masks/"

# do the function here to create a new 4D parcelation image with the
# information of the time_series combined in the 400 ROIs from schaefer


def tsv_to_4d_parcellation(tsv_path, subj_atlas, affine, header, out_path, n_rois=None):
    """
    Create a 4D NIfTI where each ROI is filled with its time series
    from a TSV (timepoints × ROIs).

    tsv_path: TSV file with time series (T × N_ROIs)
    subj_atlas: 3D np.array of ROI labels (already personalized & masked)
    affine, header: taken from the mask related to the parcellation
    out_path: output NIfTI file for 4D image
    n_rois: if not None, use first n_rois columns of TSV and
            assume labels 1..n_rois. If None, infer from max label.
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


# define the atlas files here
schaefer_atlas_file = "/media/jmm/One_Touch/UTA_Lab/ENIGMA-PTSD/Atlases/tpl-MNI152NLin2009cAsym_atlas-schaefer2011Combined_dseg.nii.gz"
power_atlas_file = "/media/jmm/One_Touch/UTA_Lab/ENIGMA-PTSD/Atlases/tpl-MNI152NLin2009cAsym_atlas-power2011_dseg.nii.gz"
brainnetome_atlas_file = "/media/jmm/One_Touch/UTA_Lab/ENIGMA-PTSD/Atlases/tpl-MNI152NLin2009cAsym_atlas-brainnetomeCombined_dseg.nii.gz"

# get here first the sites names to continue processing - as first level folders
for site in os.listdir(path_data):
    if "." not in site:
        for subjects in os.listdir(path_data + "/" + site):
            # get the brain mask for each subject
            if site == "Beijing":
                subject_mask = subjects[0:4] + f"{int(subjects[4:]) - 880:03d}"
                subject_val = subjects
            elif site == "Capetown":
                subject_mask = subjects.replace("-capetown", "").replace("-tygerberg", "")
                subject_val = subject_mask
            elif site == "Cisler":
                subject_mask = subjects.replace("_", "")
                subject_val = subject_mask.replace("D", "d").replace("P", "p")
            elif site == "UMN":
                subject_mask = subjects.replace("_", "")
                subject_val = subject_mask.replace("M", "m")
            else:
                subject_mask = subjects
                subject_val = subjects

            if "." not in subjects and not os.path.exists(
                path_data + "/" + site + "/" + subjects + f"/{subject_val}_schaefer_image.nii.gz"
            ):  # validates if the file exist and if not it can do preprocessing!!
                mask_file = (
                    masks_path + "/" + site + "/" + subject_mask + "-preproc_brain_mask.nii.gz"
                )

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

                # breakpoint()
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

                    # first personalize the evaluation for the schaefer images - leave
                    # the dilation in False and min_voxels in 20
                    subj_atlas_schaefer, mask_subj_img_schaefer = personalize_atlas_for_subject(
                        atlas_path=schaefer_atlas_file,
                        subj_mask_path=mask_file,
                        out_path=path_data
                        + "/"
                        + site
                        + "/"
                        + subjects
                        + f"/{subject_val}_schaefer_image.nii.gz",
                        min_voxels=20,
                        dilate=False,
                        dilation_iters=1,
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
                        + f"/{subject_val}_schaefer_4d_image.nii.gz",
                        n_rois=434,
                    )

                    # second personalize the evaluation for the power images - leave the dilation in False and min_voxels in 20 - SKIP power for now!!
                    # subj_atlas_power, mask_subj_img_power = personalize_atlas_for_subject(atlas_path=power_atlas_file, subj_mask_path=mask_file, out_path=path_data + '/' + site + '/' + subjects + f'/{subjects}_power_image.nii.gz', min_voxels=20, dilate=False, dilation_iters=1)
                    # tsv_to_4d_parcellation(tsv_path=path_file_power_file, subj_atlas=subj_atlas_power, affine=mask_subj_img_power.affine, header=mask_subj_img_power.header,  out_path=path_data + '/' + site + '/' + subjects + f'/{subjects}_power_4d_image.nii.gz', n_rois=264)

                    # third personalize the evaluation for the brainnetome images -
                    # leave the dilation in False and min_voxels in 20
                    subj_atlas_brainnetome, mask_subj_img_brainnetome = (
                        personalize_atlas_for_subject(
                            atlas_path=brainnetome_atlas_file,
                            subj_mask_path=mask_file,
                            out_path=path_data
                            + "/"
                            + site
                            + "/"
                            + subjects
                            + f"/{subject_val}_brainnetome_image.nii.gz",
                            min_voxels=20,
                            dilate=False,
                            dilation_iters=1,
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
                        + f"/{subject_val}_brainnetome_4d_image.nii.gz",
                        n_rois=263,
                    )

                    logger.success(f"Images from {subjects} in site {site} have been processed!!")
                else:
                    logger.error(
                        f"Mask for subject {subjects} does not exist in the folder!!.. continue..."
                    )

        logger.success(f"Images from site {site} ALL have been processed!!")
