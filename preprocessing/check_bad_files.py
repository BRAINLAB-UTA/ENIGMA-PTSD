"""
Utilities for quickly validating the integrity of ENIGMA-PTSD NIfTI outputs.

This script scans ENIGMA site/subject folders for generated 4D ROI-time-series
NIfTI files (e.g., Schaefer and Brainnetome) and performs fast integrity checks
to detect truncated/corrupted files without loading full volumes into memory.

   Typical use:
   - Run after large rsync/copy jobs or after bulk NIfTI creation.
   - Log subjects/sites where files cannot be read reliably.
   - Do python check_bad_files.py and specify the relative or absolute path to the RSData
"""

import os

import nibabel as nib
import numpy as np
from loguru import logger

path_data = "../../Data/RSData/"

ok_files = []
bad_files = []


def quick_check_nifti(path):
    """
    Fast integrity check for both .nii and .nii.gz:
    - loads header only
    - tries to read a single voxel near the end via dataobj
    - returns True if OK, False if error

    This function perform a fast integrity check for a NIfTI file (.nii or .nii.gz).

    The check is designed to be lightweight:
    1) Loads the header via nibabel with memory mapping (lazy proxy).
    2) Attempts to read a single voxel at the last index of the array.
       This forces an actual file read and typically surfaces truncation,
       compression, or I/O errors without loading the full image.

    Args:
        path: (str) Path to a NIfTI file (.nii or .nii.gz).

    Returns: (bool)
        True if the header loads and the last voxel can be accessed.
        False if nibabel fails to load the image, the shape is invalid,
        or data access triggers an exception (likely corruption/truncation)
    """
    try:
        img = nib.load(path, mmap=True)  # header + lazy proxy
    except Exception:
        return False

    shape = img.shape
    if len(shape) == 0:
        return False

    # index of the "last voxel" (or last timepoint if 4D)
    idx = tuple(s - 1 for s in shape)

    try:
        _ = img.dataobj[idx]  # triggers real file read for that slice
    except Exception:
        return False

    return True


# DO THIS VALIDATION ONLY IF YOU NEED TO CHECK PARTIAL FILE OFFSET


def check_nifti_uncompressed(path):
    """
    Quick integrity check for .nii (uncompressed) using header + file size.
    Returns True if looks OK, False if clearly truncated/damaged.
    """
    img = nib.load(path)  # header-only, cheap
    hdr = img.header

    data_shape = hdr.get_data_shape()
    dtype = hdr.get_data_dtype()
    offset = hdr.get_data_offset()

    n_vox = int(np.prod(data_shape))
    expected_bytes = int(offset + n_vox * dtype.itemsize)
    actual_bytes = os.path.getsize(path)

    breakpoint()
    # actual must be at least expected
    return actual_bytes >= expected_bytes


"""
 *** Main check bad file execution across all the RSData ***
"""

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

            fpath_schaefer = (
                path_data
                + "/"
                + site
                + "/"
                + subjects
                + f"/{subject_val}_schaefer_4d_mni_image.nii.gz"
            )  # os.path.join(dirpath, fname)
            fpath_brain = (
                path_data
                + "/"
                + site
                + "/"
                + subjects
                + f"/{subject_val}_brainnetome_4d_mni_image.nii.gz"
            )

            # do the validation if this is necessary
            val_sch = quick_check_nifti(fpath_schaefer)
            val_brain = quick_check_nifti(fpath_brain)

            logger.info(f"reading {fpath_schaefer}")

            if val_sch is False or val_brain is False:
                logger.error(f"[BAD] {fpath_schaefer} {fpath_brain}")
