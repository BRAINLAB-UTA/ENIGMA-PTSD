"""
This module generates subject-level structural ROI images in MNI space from ENIGMA Excel measures.

This script converts cortical structural measures (surface area, thickness, volume)
from ENIGMA Excel sheets into 3D NIfTI images by assigning each ROI value to the
voxels belonging to that ROI in an atlas segmentation.

Key steps:
 - Build/prepare atlas label lists (Destrieux via nilearn, DKT40 via ctab mapping).
 - Resample the atlas to a target brain mask grid and remove out-of-mask voxels.
 - For each subject row in the Excel sheet, create an ROI-valued 3D volume where
   each atlas region is filled with the subject’s measure for that region.

Outputs are organized under per-site and per-subject folders.
"""

import os
import warnings
import sys

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
    List subdirectory names in 'directory_path' normalized to lowercase.

    This is used to match site identifiers across datasets where naming/casing
    differs between filesystem folders and ENIGMA metadata.

    Args:
      directory_path: (str) Directory whose immediate subfolders will be listed.

    Returns:
      A list of subfolder names in lowercase (one level deep).
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
    Resample an integer atlas into a target mask grid and constrain labels to the mask.

    Steps:
    - Resample `atlas_path` into the voxel grid of `mni_mask_path` using nearest-neighbor.
    - Zero out labels outside the brain mask.
    - Optionally dilate each ROI within the mask to patch small holes.
    - Remove ROI fragments smaller than `min_voxels`.

    Args:
      atlas_path: Path to an integer-labeled atlas segmentation NIfTI.
      mni_mask_path: Path to a brain mask NIfTI defining output grid/space.
      min_voxels: Minimum voxel count to keep an ROI after masking.
      dilate: Whether to apply constrained dilation per ROI.
      dilation_iters: Number of dilation iterations (usually 1).

    Returns:
      atlas: 3D integer NumPy array of atlas labels in mask space.
      mask_img: Loaded mask NIfTI image for affine/header reuse.
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
    subj_atlas, subj_row, excel_columns, atlas_labels, affine, header, out_path, dkt_indices
):
    """
    Create a 3D ROI-valued NIfTI by mapping ENIGMA Excel values onto atlas regions.

    This function builds a per-ROI value vector from the subject’s Excel row and
    then fills the corresponding voxels in `subj_atlas` for each ROI.

    Two mapping modes are supported:
    -    Destrieux mode: `dkt_indices` is None; ROI names are derived from Excel headers
        and matched against `atlas_labels` (e.g., Destrieux label names).
    -  DKT40 mode: `dkt_indices` provides the label IDs present in the DKT atlas volume;
        ROI names are normalized (L_/R_ → lh_/rh_) to match `atlas_labels`.

    Args:
      subj_atlas: 3D integer label array (resampled/masked atlas in target grid).
      subj_row: Subject ROI values extracted from the Excel row (aligned with columns).
      excel_columns: Excel column names corresponding to `subj_row`.
      atlas_labels: ROI name list (DKT) or labels structure (Destrieux).
      affine: Output affine (typically from mask).
      header: Output header (typically from mask).
      out_path: Output NIfTI path to write the 3D ROI-valued image.
      dkt_indices: If not None, list of atlas label IDs for DKT ROI assignment.

    """

    out_vals = np.zeros((len(atlas_labels)), dtype=np.float32)
    out_vol = np.zeros_like(subj_atlas, dtype=np.float32)

    for index_roi in range(0, len(excel_columns)):
        # fix the string for the rest of the evaluation
        if dkt_indices is None:
            str_cmp = list(excel_columns[index_roi][1:])
            str_cmp[1] = " "
            str_cmp = "".join(str_cmp)
            str_cmp = str_cmp[0 : str_cmp.rfind("_")]
        else:
            str_cmp = excel_columns[index_roi]
            str_cmp = str_cmp.replace("L_", "lh_").replace("R_", "rh_")
            str_cmp = str_cmp[0 : str_cmp.rfind("_")]
        val_index = atlas_labels.index(str_cmp)
        out_vals[val_index] = subj_row[index_roi]
        out_vals = np.nan_to_num(out_vals, nan=0.0)

    for index_roi in range(0, len(atlas_labels)):
        if dkt_indices is None:
            roi_mask = subj_atlas == index_roi
        else:
            roi_mask = subj_atlas == dkt_indices[index_roi]
        out_vol[roi_mask] = out_vals[index_roi]

    img = nib.Nifti1Image(out_vol, affine, header)
    nib.save(img, out_path)
    logger.info(f"Saved structural ROI image → {out_path}")  # similar to the RSData


# ---------------------------------------------------------------------
# 3) MAIN WORKFLOW
# ---------------------------------------------------------------------
def run_structural_parcellation(
    excel_path,
    atlas_path,
    atlas_type,
    mni_mask_path,
    structural_path,
    measure_tag,
    atlas_labels,
    dkt_indeces,
):
    """
    Main driver to create structural ROI images for all subjects in an ENIGMA Excel sheet.

    This function:
    - Loads the Excel table indexed by subject ID.
    - Builds a site allow-list from existing RSData site folders plus manually included sites.
    - Prepares a mask-constrained atlas in the target grid (via resampling + masking).
    - Iterates over subjects and writes a subject-specific ROI-valued NIfTI per measure.

    Args:
      excel_path: Path to an ENIGMA Excel file with cortical measures and an "ID" column.
      atlas_path: Path to an atlas segmentation NIfTI (DKT40 or Destrieux/a2009s).
      atlas_type: Atlas name identifier (e.g., "DKT40" or "Destrieux").
      mni_mask_path: Path to a brain mask NIfTI defining the output grid/space.
      structural_path: Root output directory where site/subject folders are created.
      measure_tag: Measure suffix used in filenames ("surf", "thick", or "vol").
      atlas_labels: ROI label names aligned to atlas regions (format depends on atlas_type).
      dkt_indeces: DKT label IDs if using DKT40; otherwise None.
    """

    rs_data_path = "../../Data/RSData/"
    sites_in_rs = get_lowercase_subfolders(directory_path=rs_data_path)
    sites_to_include = [
        "uw_cisler",
        "uw_grupe",
        "ontario",
        "mclean_kaufman",
        "minn_va",
        "munster",
        "nanjing",
        "waco_va",
    ]
    sites_in_rs = sites_in_rs + sites_to_include

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

            # check if the path has been already created..
            # logger.info(out_path)
            if site in sites_to_include:  # not os.path.exists(out_path):
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
                        dkt_indeces,
                    )
                else:
                    excel_values_to_3d_parcellation(
                        personalized_atlas,
                        row[2:60].tolist(),
                        excel_columns[2:60],
                        atlas_labels,
                        mask_img.affine,
                        mask_img.header,
                        out_path,
                        dkt_indeces,
                    )

                logger.success(f"Subject {subj_id} processed ({measure_tag})")


"""
  ** Main code section here**
"""


# START THE MAIN CODE HER AND INVOKE THEM AS THE FUNCTION IN 3**
# get the Structural path
struct_path = str(sys.argv[1])

# use nilearn to define what will be the Destrieux atlas
destrieux = fetch_atlas_destrieux_2009(lateralized=True)

# get the DKT atlas previous downloaded
A2009S_ATLAS = destrieux.maps  # path to NIfTI file
A2009S_LABELS = destrieux.labels
DKT40_ATLAS = "./bert_aparc.DKTatlas+aseg.nii.gz"
atlas_dkt = nib.load(DKT40_ATLAS).get_fdata().astype(int)
# Extract label ids
labels_dkt = sorted(list(set(atlas_dkt.ravel())))
labels_dkt = np.asarray(labels_dkt).astype(int).tolist()

# Load ctab from annot
ctab_file = "./aparc.annot.DKTatlas.ctab"

mapping = {}

# define the unique labels from the dkt ctab
with open(ctab_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        name = parts[1]
        mapping[idx] = name

# Now produce final DKT volume → ROI name mapping:
list_labels_dkt = []
indices_dkt = []
for l in labels_dkt:
    if l >= 2000:
        hemi = "rh"
        base = l - 2000
    else:
        hemi = "lh"
        base = l - 1000

    region = mapping.get(base, "unknown")

    if region != "unknown":
        indices_dkt.append(l)
        list_labels_dkt.append(f"{hemi}_{region}")

# comment this if you consider..
# a2009s_path = api.get(
#    'MNI152NLin2009cAsym',
#    atlas='aparc.a2009s',
#    suffix='dseg',
#    resolution=1,
#    extension='.nii.gz'
# )

logger.info(f"the Destrieux atlas path is {A2009S_ATLAS}")
logger.info(f"the DKT40 atlas path is {DKT40_ATLAS}")


mni_mask_path = api.get("MNI152NLin2009cAsym", desc="brain", suffix="mask", extension=".nii.gz")

# excel files as dict array
excel_files = {
    "a2009s_surf": f"{struct_path}CorticalMeasuresENIGMA_a2009s_SurfAvg.xlsx",
    "a2009s_thick": f"{struct_path}CorticalMeasuresENIGMA_a2009s_ThickAvg.xlsx",
    "a2009s_vol": f"{struct_path}CorticalMeasuresENIGMA_a2009s_volAvg.xlsx",
    "DKT40_surf": f"{struct_path}CorticalMeasuresENIGMA_DKT40_SurfAvg.xlsx",
    "DKT40_thick": f"{struct_path}CorticalMeasuresENIGMA_DKT40_ThickAvg.xlsx",
    "DKT40_vol": f"{struct_path}CorticalMeasuresENIGMA_DKT40_volAvg.xlsx",
}

# read the directories in the dict array
for tag, excel_path in excel_files.items():
    atlas_type = "DKT40" if "DKT40" in tag else "Destrieux"
    atlas_path = DKT40_ATLAS if atlas_type == "DKT40" else A2009S_ATLAS

    measure = tag.split("_")[1]  # surf / thick / vol

    if atlas_type == "Destrieux":
        atlas_LABELS = A2009S_LABELS
        dkt_indices = None
    else:
        atlas_LABELS = list_labels_dkt
        dkt_indices = indices_dkt

    run_structural_parcellation(
        excel_path=excel_path,
        atlas_path=atlas_path,
        atlas_type=atlas_type,
        mni_mask_path=str(mni_mask_path[0]),
        structural_path=struct_path,
        measure_tag=measure,
        atlas_labels=atlas_LABELS,
        dkt_indeces=dkt_indices,
    )
