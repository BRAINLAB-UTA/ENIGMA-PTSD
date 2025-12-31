"""
Build site/subject-indexed NumPy tensors for fALFF and ReHo 3D images (and optional RS 4D volumes).

This script provides utilities to:
- Validate NIfTI integrity quickly (detect truncated/corrupted .nii/.nii.gz).
-   Load many 3D or 4D NIfTI files into a single NumPy array or disk-backed memmap,
  keeping subject order consistent.
- Read fALFF/ReHo derivatives per ENIGMA site, assemble arrays, and serialize
  them for fast reuse in downstream dataloaders.

Designed for large ENIGMA datasets where repeated NIfTI I/O is expensive.
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
falff_reho_folder = "../../Data/falff_reho/"


def quick_check_nifti(path):
    """
    Quickly validate that a NIfTI file can be read without fully loading it.

    The check is intentionally lightweight:
      - Load header + lazy data proxy (mmap/lazy loading).
      - Attempt to access a voxel near the end of the array via 'dataobj'
        to trigger an actual read from disk.

    Args:
      path: Path to a .nii or .nii.gz file.

    Returns:
      True if the file header loads and voxel access succeeds.
      False if loading or voxel access raises an exception (likely corruption).
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


def load_4d_tensor(file_map, dtype=np.float32, memmap_path=None, atlas=None):
    """
    Load multiple 4D NIfTI images into a single 5D tensor with consistent subject ordering.

    Files are specified via `file_map` and loaded in sorted-key order so the output
    tensor axis 0 is stable and reproducible.

    Args:
      file_map: Mapping {key -> nifti_path}. Keys are sortable (e.g., subject IDs).
      dtype: Floating dtype used when materializing data.
      memmap_path: If provided, write into a disk-backed memmap array at this path.
      atlas: Optional tag used only for logging/context.

    Returns:
      tensor: Array of shape (N, X, Y, Z, T) (or a memmap with that shape).
      keys_sorted: Sorted list of keys matching axis 0 of `tensor`.
      affine: Affine from the first loaded NIfTI.
      header: Header from the first loaded NIfTI.

    Raises:
      ValueError: If 'file_map' is empty or the first file is not 4D.
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
    Load multiple 3D NIfTI images into a single 4D tensor with consistent subject ordering.

    This is the 3D analogue of 'load_4d_tensor', used for scalar maps such as
    fALFF or ReHo, expected to be shaped (X, Y, Z) per subject.

    Args:
      file_map: Mapping {key -> nifti_path}. Keys are sortable (e.g., subject IDs).
      dtype: Floating dtype used when materializing data.
      memmap_path: If provided, write into a disk-backed memmap array at this path.
      atlas: Optional tag used only for logging/context.

    Returns:
      tensor: Array of shape (N, X, Y, Z) (or a memmap with that shape).
      keys_sorted: Sorted list of keys matching axis 0 of `tensor`.
      affine: Affine from the first loaded NIfTI.
      header: Header from the first loaded NIfTI.

    Raises:
      ValueError: If `file_map` is empty or the first file is not 3D.
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


def load_falff_reho(
    metric_map, mask_map, dtype=np.float32, memmap_path=None, atlas=None
):
    """
     Load fALFF and ReHo datasets into aligned subject-major tensors.

     This helper loads two 3D image collections (fALFF and ReHo) while preserving
     consistent ordering across both modalities.

     Args:
       file_map_falff: Mapping {key -> fALFF nifti_path}.
       file_map_reho: Mapping {key -> ReHo nifti_path}.
       dtype: Floating dtype used when materializing data.
       memmap_dir: Optional directory where memmaps for each modality are created.

    Returns:
       falff_tensor: (N, X, Y, Z) subject-major array/memmap.
       reho_tensor: (N, X, Y, Z) subject-major array/memmap.
       keys_sorted: Sorted subject keys used for both tensors.
       affine: Affine from the first loaded file (assumed consistent).
       header: Header from the first loaded file (assumed consistent).

    Raises:
       ValueError: If subject keys do not match between modalities.
    """

    keys_sorted = sorted(metric_map.keys())
    n_subj = len(keys_sorted)

    # First subject → define reference mask and voxel count
    first_key = keys_sorted[0]

    metric_img = nib.load(metric_map[first_key], mmap=True)
    metric_data = metric_img.get_fdata(dtype=dtype)

    mask_img = nib.load(mask_map[first_key], mmap=True)
    mask_data = mask_img.get_fdata().astype(bool)

    if metric_data.shape != mask_data.shape:
        raise ValueError(f"Mask/metric shape mismatch for {first_key}")

    ref_mask = mask_data  # you can also intersect masks across subjects if you want
    n_vox = int(ref_mask.sum())

    if memmap_path is not None:
        tensor = np.memmap(
            memmap_path,
            mode="w+",
            shape=(n_subj, ref_mask.shape[0], ref_mask.shape[1], ref_mask.shape[2]),
            dtype=dtype,
        )
    else:
        tensor = np.empty((n_subj, n_vox), dtype=dtype)

    # fill first row
    vol_falff = np.zeros(ref_mask.shape, dtype=np.float32)
    vol_falff[ref_mask] = metric_data[ref_mask]

    tensor[0] = vol_falff

    affine = metric_img.affine
    header = metric_img.header

    for i, key in enumerate(keys_sorted[1:], start=1):
        good_or_bad = quick_check_nifti(metric_map[key])
        if good_or_bad is False:
            continue

        metric_img = nib.load(metric_map[key], mmap=True)
        metric_data = metric_img.get_fdata(dtype=dtype)

        mask_img = nib.load(mask_map[key], mmap=True)
        mask_data = mask_img.get_fdata().astype(bool)

        if metric_data.shape != ref_mask.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {metric_data.shape} != {ref_mask.shape}"
            )

        vol_falff_subj = np.zeros(ref_mask.shape, dtype=np.float32)
        vol_falff_subj[ref_mask] = metric_data[ref_mask]
        vol_falff_subj[~mask_data] = 0.0

        # you can either:
        #  - use ref_mask only, or
        #  - combine: valid_mask = ref_mask & mask_data
        # Here we stick to ref_mask to keep voxel indices consistent:
        tensor[i] = vol_falff_subj  # metric_data[ref_mask]

        logger.info(
            f"processing and saving numpy for subject {key} 3D images for atlas {atlas}"
        )

    return tensor, keys_sorted, affine, header


def read_files_outer_loop(path_dir: str, type_data: str, sites: list, subjects: list):
    """
    - get the path_dir as string from the main folder to read all the
    sites inside
    - type_data as str to define what type of data you are reading
    - read the modality suffix as str depending what type of connectome or structural representation
      you want to read.
    - sites list: read the sites this order if they  are parsed different than None
    - subjects: read the subjects included on each site directory in order as a list of lists
    """

    sites_agg = []
    SUB = []

    # read the folder inside RSData first and generate the numpy for each of it
    if sites is None and subjects is None:
        for site in os.listdir(path_dir):
            if "." not in site:
                # initialize subjects empty
                subject_vals = []

                if site == "old_sites":  # or site != "Michigan": #site in done_sites:
                    continue

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
                elif type_data == "falff_reho_data":
                    alff_3d_map = {}
                    alff_mask_map = {}
                    falff_3d_map = {}
                    falff_mask_map = {}
                    reho_3d_map = {}
                    reho_mask_map = {}

                if site == "QC_reports":  # skip QC reports
                    continue

                if type_data == "falff_reho_data":
                    site_whole = site + "/" + site
                else:
                    site_whole = site

                for subjects in os.listdir(path_dir + "/" + site_whole):
                    # read here the information necessary to read the folder path
                    if type_data == "falff_reho_data":
                        pattern = os.path.join(
                            "/".join([path_dir, site, site, subjects]), "*.nii.gz"
                        )
                    else:
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

                        # define here the falff_reho_modality
                        if type_data == "falff_reho_data":
                            # data , _, _ = load_file_nib(nii_file)
                            if "-fALFF_alff" in nii_file:
                                alff_3d_map[(site, subjects)] = nii_file
                                alff_mask_map[(site, subjects)] = nii_file.replace(
                                    "_alff.", "_mask."
                                )
                            elif "-fALFF_falff" in nii_file:
                                falff_3d_map[(site, subjects)] = nii_file
                                falff_mask_map[(site, subjects)] = nii_file.replace(
                                    "_falff.", "_mask."
                                )
                            elif "-reHo_reho" in nii_file:
                                reho_3d_map[(site, subjects)] = nii_file
                                reho_mask_map[(site, subjects)] = nii_file.replace(
                                    "_reho.", "_mask."
                                )

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
                elif type_data == "falff_reho_data":
                    alff_tensor_3d, _, _, _ = load_falff_reho(
                        alff_3d_map,
                        alff_mask_map,
                        dtype=np.float32,
                        memmap_path=f"./rs_alff_{site}.dat",
                        atlas="alff",
                    )
                    falff_tensor_3d, _, _, _ = load_falff_reho(
                        falff_3d_map,
                        falff_mask_map,
                        dtype=np.float32,
                        memmap_path=f"./rs_falff_{site}.dat",
                        atlas="falff",
                    )
                    reho_tensor_3d, subject_sorted, _, _ = load_falff_reho(
                        reho_3d_map,
                        reho_mask_map,
                        dtype=np.float32,
                        memmap_path=f"./rs_reho_{site}.dat",
                        atlas="reho",
                    )

                # define the aggregate sites for setting the tensor
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
                    # DATA.append([rs_tensor_3d_schaefer, rs_tensor_4d_schaefer, rs_tensor_3d_brainnetome, rs_tensor_4d_brainnetome])
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

                elif type_data == "falff_reho_data":
                    np.savez_compressed(
                        f"../../Data/{type_data}_{site}_data.npz",
                        alff_tensor_3d=alff_tensor_3d,
                        falff_tensor_3d=falff_tensor_3d,
                        reho_tensor_3d=reho_tensor_3d,
                        subjects=np.array(subject_sorted),
                    )

                # Save the arrays as numpys here to avoid high overhead in the data
                logger.success(f"site {site} has been saved in the pkl file..")
                os.remove(f"./rs_alff_{site}.dat")
                os.remove(f"./rs_falff_{site}.dat")
                os.remove(f"./rs_reho_{site}.dat")
                os.remove(f"./rs_surf_dkt_{site}.dat")
                os.remove(f"./rs_vol_dkt_{site}.dat")
                os.remove(f"./rs_thick_dkt_{site}.dat")
                os.remove(f"./rs_surf_dest_{site}.dat")
                os.remove(f"./rs_vol_dest_{site}.dat")
                os.remove(f"./rs_thick_dest_{site}.dat")

    # return the aggregated data and sites list per modality
    return sites_agg, SUB


"""
  **Main section of the code here**
"""

# main part of the code here
sites_all, SUB = read_files_outer_loop(
    path_dir=RSdata_folder, type_data="rs_data", sites=None, subjects=None
)
sites_all, SUB = read_files_outer_loop(
    path_dir=Structural_folder, type_data="structural_data", sites=None, subjects=None
)
sites_all, SUB = read_files_outer_loop(
    path_dir=falff_reho_folder, type_data="falff_reho_data", sites=None, subjects=None
)
