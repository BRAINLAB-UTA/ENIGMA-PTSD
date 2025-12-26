import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger
from nilearn.datasets import fetch_atlas_destrieux_2009
from nilearn.image import resample_to_img
from scipy.ndimage import binary_dilation
from templateflow import api

# To ignore all warnings:
warnings.filterwarnings("ignore")
# To ignore specific categories of warnings (e.g., DeprecationWarning):
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_lowercase_subfolders(directory_path):
    """
    Retrieves a list of all subfolder names within a given directory,
    with all names converted to lowercase.

    Args:
        directory_path (str): The path to the directory to scan.

    Returns:
        list: A list of lowercase subfolder names.
    """
    # get lower case strings from subfolders
    subfolders = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item.lower())

    return subfolders


# this code will be similar to the one applied to the RSData but for the structural data
# ---------------------------------------------------------------------
# 1) PERSONALIZE ATLAS TO SUBJECT MASK (copied from RS script)
# ---------------------------------------------------------------------


def personalize_atlas_for_subject(
    atlas_path, mni_mask_path, min_voxels=10, dilate=False, dilation_iters=1
):
    """
    Same logic as in the RS script:
    - Resample atlas to subject mask grid
    - Mask out voxels outside brain
    - (optional) small dilation to fix small gaps
    - Remove tiny fragments (<min_voxels)
    """
    atlas_img = nib.load(atlas_path)
    mask_img = nib.load(mni_mask_path)

    # Resample to subject mask
    atlas_resampled = resample_to_img(atlas_img, mask_img, interpolation="nearest")

    atlas = atlas_resampled.get_fdata().astype(int)
    mask = mask_img.get_fdata() > 0
    atlas[~mask] = 0

    # Optional dilation
    if dilate:
        new_atlas = atlas.copy()
        _roi_labels = np.unique(atlas)
        for label in _roi_labels[_roi_labels != 0]:
            region = atlas == label
            if region.sum() == 0:
                continue
            dil = binary_dilation(region, iterations=dilation_iters)
            dil = dil & mask
            new_atlas[dil] = label
        atlas = new_atlas

    # Drop small regions
    labels = np.unique(atlas)
    for label in labels[labels != 0]:
        if (atlas == label).sum() < min_voxels:
            atlas[atlas == label] = 0

    nib.Nifti1Image(atlas, mask_img.affine, mask_img.header)
    # nib.save(out_img, out_path)
    logger.info("Personalized image has been processed!!")

    return atlas, mask_img


# ---------------------------------------------------------------------
# 2) CREATE STRUCTURAL PARRCELLATION NIfTI
# ---------------------------------------------------------------------
def excel_values_to_3d_parcellation(
    subj_atlas, subj_row, excel_columns, atlas_labels, affine, header, out_path
):
    """
    Create a 3D NIfTI where each voxel is assigned a structural metric
    (using the values in subj_row).

    subj_atlas: 3D array of ROI labels
    label_mapping: dict {label: ROI_column_name}
    subj_row: pandas Series for that subject
    affine/header: from personalized atlas
    """

    out_vals = np.zeros((len(atlas_labels)), dtype=np.float32)
    out_vol = np.zeros_like(subj_atlas, dtype=np.float32)

    for index_roi in range(0, len(excel_columns)):
        # fix the string for the rest of the evaluation
        str_cmp = list(excel_columns[index_roi][1:])
        str_cmp[1] = " "
        str_cmp = "".join(str_cmp)
        str_cmp = str_cmp[0 : str_cmp.rfind("_")]
        val_index = atlas_labels.index(str_cmp)
        out_vals[val_index] = subj_row[index_roi]
        out_vals = np.nan_to_num(out_vals, nan=0.0)

    for index_roi in range(0, len(atlas_labels)):
        roi_mask = subj_atlas == index_roi
        out_vol[roi_mask] = out_vals[index_roi]

    img = nib.Nifti1Image(out_vol, affine, header)
    nib.save(img, out_path)
    logger.info(f"Saved structural ROI image → {out_path}")  # similar to the RSData


# ---------------------------------------------------------------------
# 3) MAIN WORKFLOW
# ---------------------------------------------------------------------
def run_structural_parcellation(
    excel_path, atlas_path, atlas_type, mni_mask_path, structural_path, measure_tag, atlas_labels
):
    """
    excel_path: ENIGMA Excel file (SurfAvg / ThickAvg / VolAvg)
    atlas_path: DKT40 / a2009s segmentation NIfTI
    atlas_type: "DKT40" or "Destrieux"
    mni_mask_path: generalized MNI mask
    structural_path: give the suffix for creating dirs
    measure_tag: "surf", "thick", "vol",
    atlas_labels: list of the labels inside the atlas masks
    """

    rs_data_path = "../Data/RSData/"
    sites_in_rs = get_lowercase_subfolders(directory_path=rs_data_path)

    # Load Excel
    df = pd.read_excel(excel_path)
    df = df.set_index("ID")  # IMPORTANT — ENIGMA uses "ID" column
    excel_columns = df.columns.tolist()

    # Personalize atlas
    personalized_atlas, mask_img = personalize_atlas_for_subject(
        atlas_path, mni_mask_path, min_voxels=20, dilate=False
    )

    # Process each subject
    for subj_id, row in df.iterrows():
        # define here the site here
        site = row[0]

        if site in sites_in_rs:
            path_site = structural_path + str(site)
            if not os.path.exists(path_site):
                os.makedirs(path_site, exist_ok=True)
            if not os.path.exists(path_site + "/sub-" + str(subj_id)):
                os.makedirs(path_site + "/sub-" + str(subj_id), exist_ok=True)

            subject_path = path_site + "/sub-" + str(subj_id)
            out_path = os.path.join(
                subject_path, f"{subj_id}_{atlas_type}_{measure_tag}_struct3D.nii.gz"
            )

            # create the customized image here
            if atlas_type == "Destrieux":
                excel_values_to_3d_parcellation(
                    personalized_atlas,
                    row[2:150].tolist(),
                    excel_columns[2:150],
                    atlas_labels["name"].tolist(),
                    mask_img.affine,
                    mask_img.header,
                    out_path,
                )

                logger.success(f"Subject {subj_id} processed ({measure_tag})")


# START THE MAIN CODE HER AND INVOKE THEM AS THE FUNCTION IN 3**
# get the Structural path
struct_path = "../Data/Structural/"

# use nilearn to define what will be the Destrieux atlas
destrieux = fetch_atlas_destrieux_2009(lateralized=True)

A2009S_ATLAS = destrieux.maps  # path to NIfTI file
A2009S_LABELS = destrieux.labels

logger.info(f"the Destrieux atlas path is {A2009S_ATLAS}")

mni_mask_path = api.get("MNI152NLin2009cAsym", desc="brain", suffix="mask", extension=".nii.gz")

# Check this tomorrow using maybe FreeSurfer
# excel files as dict array
# excel_files = {
#        "DKT40_surf": f"{struct_path}CorticalMeasuresENIGMA_DKT40_SurfAvg.xlsx",
#        "DKT40_thick": f"{struct_path}CorticalMeasuresENIGMA_DKT40_ThickAvg.xlsx",
#        "DKT40_vol": f"{struct_path}CorticalMeasuresENIGMA_DKT40_volAvg.xlsx",
#        "a2009s_surf": f"{struct_path}CorticalMeasuresENIGMA_a2009s_SurfAvg.xlsx",
#        "a2009s_thick": f"{struct_path}CorticalMeasuresENIGMA_a2009s_ThickAvg.xlsx",
#        "a2009s_vol": f"{struct_path}CorticalMeasuresENIGMA_a2009s_volAvg.xlsx",
# }

excel_files = {
    "a2009s_surf": f"{struct_path}CorticalMeasuresENIGMA_a2009s_SurfAvg.xlsx",
    "a2009s_thick": f"{struct_path}CorticalMeasuresENIGMA_a2009s_ThickAvg.xlsx",
    "a2009s_vol": f"{struct_path}CorticalMeasuresENIGMA_a2009s_volAvg.xlsx",
}

# read the directories in the dict array
for tag, excel_path in excel_files.items():
    atlas_type = "DKT40" if "DKT40" in tag else "Destrieux"
    atlas_path = DKT40_ATLAS if atlas_type == "DKT40" else A2009S_ATLAS

    measure = tag.split("_")[1]  # surf / thick / vol

    if measure == "thick":
        # run the structural representation here..
        run_structural_parcellation(
            excel_path=excel_path,
            atlas_path=atlas_path,
            atlas_type=atlas_type,
            mni_mask_path=str(mni_mask_path[0]),
            structural_path=struct_path,
            measure_tag=measure,
            atlas_labels=A2009S_LABELS,
        )
