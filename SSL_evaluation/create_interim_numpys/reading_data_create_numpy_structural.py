"""
This module creates cached NumPy containers for subject-level structural ROI images.

This script loads structural ROI-valued NIfTI images (e.g., surface/thickness/volume
maps encoded into atlas space) into subject-major arrays to avoid repeated NIfTI I/O.

It includes optional upsampling/resampling logic to normalize images to a target grid
(e.g., 193×229×193) when site outputs differ in resolution.
"""

import glob
import os
import warnings

import nibabel as nib
import numpy as np
from loguru import logger
from nilearn.image import resample_img

# To ignore all warnings:
warnings.filterwarnings("ignore")
# To ignore specific categories of warnings (e.g., DeprecationWarning):
warnings.filterwarnings("ignore", category=DeprecationWarning)

RSdata_folder = "../../Data/RSData/"
Structural_folder = "../../Data/Structural/"


def load_4d_tensor(file_map, dtype=np.float32, memmap_path=None, atlas=None):
    """
    This fuction Load multiple (nominally) 4D images into a single subject-major 5D tensor and normalize shape.

    The loader:
    - Loads files in sorted key order for reproducible subject indexing.
    - Checks spatial dimensions; if an image does not match the target spatial size,
      it is resampled/upsampled to the target grid using nilearn’s resampling.

    Args:
      file_map: Mapping {key -> nifti_path}. Keys must be sortable.
      dtype: Floating dtype used when materializing data.
      memmap_path: Optional path for a disk-backed output array.
      atlas: Optional tag used only for logging/context.

    Returns:
      tensor: (N, X, Y, Z, C_or_T) array/memmap depending on stored image dimensionality.
      keys_sorted: Sorted keys aligned with tensor axis 0.
      affine: Affine from the first loaded (or target) image.
      header: Header from the first loaded (or target) image.

    Raises:
      ValueError: If no files are provided or dimensionality is unexpected.
    """

    keys_sorted = sorted(file_map.keys())
    n_subj = len(keys_sorted)

    if n_subj == 0:
        raise ValueError("No subjects in file_map")

    # Load first subject to get shape
    first_img = nib.load(file_map[keys_sorted[0]])
    first_data = first_img.get_fdata(dtype=dtype)
    if first_data.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {first_data.shape}")

    X, Y, Z, T = first_data.shape
    logger.info(f"RS shape: (X,Y,Z,T)=({X},{Y},{Z},{T}), N={n_subj}")

    # Preallocate array or memmap
    if memmap_path is not None:
        logger.info(f"Creating memmap at {memmap_path}")
        tensor = np.memmap(
            memmap_path, mode="w+", shape=(n_subj, X, Y, Z, T), dtype=dtype
        )
    else:
        tensor = np.empty((n_subj, X, Y, Z, T), dtype=dtype)

    # Fill
    tensor[0] = first_data
    affine = first_img.affine
    header = first_img.header

    for i, key in enumerate(keys_sorted[1:], start=1):
        path = file_map[key]
        img = nib.load(path)
        data = img.get_fdata(dtype=dtype)  # directly float32
        if data.shape[0] != 193:
            orig_shape = np.array(img.shape[:3], dtype=int)
            new_shape = (np.array([193, 229, 193], dtype=int)).astype(
                int
            )  # [X*2, Y*2, Z*2]
            new_affine = affine.copy()
            new_affine[:3, :3] /= 2  # voxel size / 2  => 2x resolution
            img_up = resample_img(
                img,
                target_affine=new_affine,
                target_shape=new_shape,
                interpolation="continuous",
            )

            data = img_up.get_fdata(dtype=dtype)
            affine = new_affine.copy()
            logger.info(
                f"upsampling image for subject {key} shape of image is {orig_shape}"
            )

        tensor[i] = data
        logger.info(
            f"processing and saving numpy for subject {key} 4D images for atlas {atlas}"
        )

        if i % 10 == 0 or i == n_subj - 1:
            logger.info(f"Loaded {i + 1}/{n_subj} RS 4D images")

    return tensor, keys_sorted, affine, header


def load_3d_tensor(file_map, dtype=np.float32, memmap_path=None, atlas=None):
    """
    Load multiple 3D structural images into a subject-major 4D tensor and normalize shape.

    Args:
       file_map: Mapping {key -> nifti_path}. Keys must be sortable.
       dtype: Floating dtype used when materializing data.
       memmap_path: Optional path for a disk-backed output array.
       atlas: Optional tag used only for logging/context.

    Returns:
     tensor: (N, X, Y, Z) array/memmap.
     keys_sorted: Sorted keys aligned with tensor axis 0.
     affine: Affine from the first loaded (or target) image.
     header: Header from the first loaded (or target) image.
    """

    keys_sorted = sorted(file_map.keys())
    n_subj = len(keys_sorted)

    first_img = nib.load(file_map[keys_sorted[0]])
    first_data = first_img.get_fdata(dtype=dtype)
    if first_data.ndim != 3:
        raise ValueError(f"Expected 3D data, got {first_data.shape}")
    X, Y, Z = first_data.shape

    if memmap_path is not None:
        tensor = np.memmap(memmap_path, mode="w+", shape=(n_subj, X, Y, Z), dtype=dtype)

    else:
        tensor = np.empty((n_subj, X, Y, Z), dtype=dtype)

    tensor[0] = first_data
    affine, header = first_img.affine, first_img.header

    for i, key in enumerate(keys_sorted[1:], start=1):
        img = nib.load(file_map[key])
        data = img.get_fdata(dtype=dtype)
        if data.shape[0] != 193:
            orig_shape = np.array(img.shape[:3], dtype=int)
            new_shape = (np.array([193, 229, 193], dtype=int)).astype(
                int
            )  # [X*2, Y*2, Z*2]
            new_affine = affine.copy()
            new_affine[:3, :3] /= 2  # voxel size / 2  => 2x resolution
            img_up = resample_img(
                img,
                target_affine=new_affine,
                target_shape=new_shape,
                interpolation="nearest",
            )

            data = img_up.get_fdata(dtype=dtype)
            affine = new_affine.copy()
            logger.info(
                f"upsampling image for subject {key} shape of image is {orig_shape}"
            )

        tensor[i] = data
        logger.info(
            f"processing and saving numpy for subject {key} 3D images for atlas {atlas}"
        )

    return tensor, keys_sorted, affine, header


def read_files_outer_loop(path_dir: str, type_data: str, sites: list, subjects: list):
    """
    Top-level structural data crawler that builds cached arrays across sites/subjects.

    The workflow typically:
    - Iterates across ENIGMA sites and subject folders.
    - Collects structural ROI-image paths for one or more measures.
    - Loads images into subject-major arrays (optionally via memmap) and persists
     them to disk (e.g., NPZ/Pickle) for fast reuse.

    Returns:
       None. Outputs are written to disk as cached containers.
    """

    sites_agg = []
    SUB = []

    sites_include = [
        "Mclean",
        "Grupe",
        "Lawson",
        "Grupe",
        "Cisler",
        "Muenster",
        "NanjingYixing",
        "WacoVA",
        "MinnVA",
    ]
    # read the folder inside RSData first and generate the numpy for each of it
    if sites is None and subjects is None:
        for site in os.listdir(path_dir):
            if "." not in site:
                # initialize subjects empty
                subject_vals = []

                # evalute the type of data inside the initial for-loop
                if type_data == "rs_data":
                    schaefer_3d_map = {}
                    brainnetome_3d_map = {}
                    schaefer_4d_map = {}
                    brainnetome_4d_map = {}
                elif type_data == "structural_data":
                    surf_3d_map_dkt = {}
                    vol_3d_map_dkt = {}
                    thick_3d_map_dkt = {}
                    surf_3d_map_dest = {}
                    vol_3d_map_dest = {}
                    thick_3d_map_dest = {}

                if site == "QC_reports" or site not in sites_include:  # skip QC reports
                    continue

                for subjects in os.listdir(path_dir + "/" + site):
                    # read here the information necessary to read the folder path
                    pattern = os.path.join(
                        "/".join([path_dir, site, subjects]), "*.nii.gz"
                    )

                    # here the pattern to read the modality type
                    nii_gz_files = glob.glob(pattern)

                    for nii_file in nii_gz_files:
                        if type_data == "rs_data":
                            if "mni" in nii_file:
                                # data , _, _ = load_file_nib(nii_file)
                                if "schaefer_mni_image" in nii_file:
                                    schaefer_3d_map[(site, subjects)] = nii_file
                                elif "brainnetome_mni_image" in nii_file:
                                    brainnetome_3d_map[(site, subjects)] = nii_file
                                elif "schaefer_4d_mni" in nii_file:
                                    schaefer_4d_map[(site, subjects)] = nii_file
                                elif "brainnetome_4d_mni" in nii_file:
                                    brainnetome_4d_map[(site, subjects)] = nii_file

                        # define the structural data here
                        if type_data == "structural_data":
                            if "DKT40_surf" in nii_file:
                                surf_3d_map_dkt[(site, subjects)] = nii_file
                            elif "DKT40_vol" in nii_file:
                                vol_3d_map_dkt[(site, subjects)] = nii_file
                            elif "DKT40_thick" in nii_file:
                                thick_3d_map_dkt[(site, subjects)] = nii_file
                            elif "Destrieux_surf" in nii_file:
                                surf_3d_map_dest[(site, subjects)] = nii_file
                            elif "Destrieux_vol" in nii_file:
                                vol_3d_map_dest[(site, subjects)] = nii_file
                            elif "Destrieux_thick" in nii_file:
                                thick_3d_map_dest[(site, subjects)] = nii_file

                    subject_vals.append(subjects)
                    # add the information all the modalities

                logger.info(f"saving numpys for site {site}!!")
                if type_data == "rs_data":
                    rs_tensor_3d_schaefer, _, _, _ = load_3d_tensor(
                        schaefer_3d_map,
                        dtype=np.float32,
                        memmap_path=f"./rs_schaefer3d_float32_{site}.dat",
                        atlas="schaefer",
                    )
                    rs_tensor_4d_schaefer, _, _, _ = load_4d_tensor(
                        schaefer_4d_map,
                        dtype=np.float32,
                        memmap_path=f"./rs_schaefer4d_float32_{site}.dat",
                        atlas="schaefer",
                    )
                    rs_tensor_3d_brainnetome, _, _, _ = load_3d_tensor(
                        brainnetome_3d_map,
                        dtype=np.float32,
                        memmap_path=f"./rs_brainnetome3d_float32_{site}.dat",
                        atlas="brainnetome",
                    )
                    rs_tensor_4d_brainnetome, _, _, _ = load_4d_tensor(
                        brainnetome_4d_map,
                        dtype=np.float32,
                        memmap_path=f"./rs_brainnetome4d_float32_{site}.dat",
                        atlas="brainnetome",
                    )
                    # data_per_site.append([schaefer_3d, brainnetome_3d, schaefer_4d, brainnetome_4d])
                    # elif type_data == "structural_data":
                elif type_data == "structural_data":
                    rs_tensor_3d_dkt_surf, _, _, _ = load_3d_tensor(
                        surf_3d_map_dkt,
                        dtype=np.float32,
                        memmap_path=f"./rs_surf_dkt_{site}.dat",
                        atlas="surf_dkt",
                    )
                    rs_tensor_3d_dkt_vol, _, _, _ = load_3d_tensor(
                        vol_3d_map_dkt,
                        dtype=np.float32,
                        memmap_path=f"./rs_vol_dkt_{site}.dat",
                        atlas="vol_dkt",
                    )
                    rs_tensor_3d_dkt_thick, _, _, _ = load_3d_tensor(
                        thick_3d_map_dkt,
                        dtype=np.float32,
                        memmap_path=f"./rs_thick_dkt_{site}.dat",
                        atlas="thick_dkt",
                    )
                    rs_tensor_3d_dest_surf, _, _, _ = load_3d_tensor(
                        surf_3d_map_dest,
                        dtype=np.float32,
                        memmap_path=f"./rs_surf_dest_{site}.dat",
                        atlas="surf_dest",
                    )
                    rs_tensor_3d_dest_vol, _, _, _ = load_3d_tensor(
                        vol_3d_map_dest,
                        dtype=np.float32,
                        memmap_path=f"./rs_vol_dest_{site}.dat",
                        atlas="vol_dest",
                    )
                    rs_tensor_3d_dest_thick, subject_sorted, _, _ = load_3d_tensor(
                        thick_3d_map_dest,
                        dtype=np.float32,
                        memmap_path=f"./rs_thick_dest_{site}.dat",
                        atlas="thick_dest",
                    )

                sites_agg.append(site)
                SUB.append(subject_sorted)
                # save the nifti files interim here if necessary
                [s for sublist in SUB for s in sublist]
                # save the nifti files interim here if necessary
                if type_data == "rs_data":
                    np.savez_compressed(
                        f"../../Data/{type_data}_{site}_data.npz",
                        rs_tensor_3d_schaefer=rs_tensor_3d_schaefer,
                        rs_tensor_4d_schaefer=rs_tensor_4d_schaefer,
                        rs_tensor_3d_brainnetome=rs_tensor_3d_brainnetome,
                        rs_tensor_4d_brainnetome=rs_tensor_4d_brainnetome,
                        subjects=np.array(subject_sorted),
                    )

                elif type_data == "structural_data":
                    np.savez_compressed(
                        f"../../Data/{type_data}_{site}_data.npz",
                        rs_tensor_3d_dkt_surf=rs_tensor_3d_dkt_surf,
                        rs_tensor_3d_dkt_vol=rs_tensor_3d_dkt_vol,
                        rs_tensor_3d_dkt_thick=rs_tensor_3d_dkt_thick,
                        rs_tensor_3d_dest_surf=rs_tensor_3d_dest_surf,
                        rs_tensor_3d_dest_vol=rs_tensor_3d_dest_vol,
                        rs_tensor_3d_dest_thick=rs_tensor_3d_dest_thick,
                        subjects=np.array(subject_sorted),
                    )

                # Save the arrays as numpys here to avoid high overhead in the data
                logger.success(f"site {site} has been saved in the pkl file..")
                os.remove(f"./rs_surf_dkt_{site}.dat")
                os.remove(f"./rs_vol_dkt_{site}.dat")
                os.remove(f"./rs_thick_dkt_{site}.dat")
                os.remove(f"./rs_surf_dest_{site}.dat")
                os.remove(f"./rs_vol_dest_{site}.dat")
                os.remove(f"./rs_thick_dest_{site}.dat")

    # return the aggregated data and sites list per modality
    return sites_agg, SUB


"""
  *** Main part of the code here ***
"""
sites_all, SUB = read_files_outer_loop(
    path_dir=Structural_folder, type_data="structural_data", sites=None, subjects=None
)
