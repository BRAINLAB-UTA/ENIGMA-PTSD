"""
This module set Dataset and DataLoader construction for ENIGMA multimodal training/evaluation
in torch.

This module defines:
- A dataset that synchronizes multiple modalities per subject:
   1) Resting-state ROI-parcellated 4D volumes (multiple atlases supported)
   2) Structural NPZ tensors (e.g., surface/thickness/volume channels)
   3) fALFF/ReHo NPZ tensors
- A DataLoader builder that returns a ready-to-train PyTorch DataLoader.

It includes site-specific normalization rules to reconcile inconsistent subject
naming conventions across ENIGMA sites and modalities.
"""

import os
import pickle
import time
import re
import warnings
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import torch
from ansi2html import Ansi2HTMLConverter
from loguru import logger
from torch.utils.data import DataLoader, Dataset

# define this to skip unexistent data
from torch.utils.data.dataloader import default_collate

# set a new renice here
try:
    # negative = higher priority (usually requires sudo)
    os.nice(-20)
except PermissionError as e:
    logger.error("No permission to raise priority (need sudo/capabilities).", e)

# set the OS environment here for renicing**
os.sched_setaffinity(0, set(range(0, 23)))
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "24"

# set the cudas here
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# To ignore all warnings:
warnings.filterwarnings("ignore")
# To ignore specific categories of warnings (e.g., DeprecationWarning):
warnings.filterwarnings("ignore", category=DeprecationWarning)

FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.level("SUCCESS", color="<green>")
logger.level("INFO", color="<white>")
logger.level("ERROR", color="<red>")

logger.add("report_final_excluded_subjects.log", format=FMT, colorize=True)


# define here the paths for reading the data coming for each subject and site
RSdata_folder = "../../../Data/RSData/"
npz_folder = "../../../Data/npz"
npy_folder = "../../../Data/npy"

structural_npz_folder = f"{npz_folder}/structural"
falff_reho_npz_folder = f"{npz_folder}/falff_reho_3d"

subject_indices_current_data = "../../../Data/npz/subjects_overlaped_all_modalities.npz"

DATA_structural = []
DATA_falff_reho = []
SITES = []
SUB = []

# plot the histogram with defined values

# define a function to skip the files that doesnt exist in the RSData part
def plot_histogram(data, x_string: str, y_string: str, bins: int, counts_show: bool):
    """
      Plot a normalized histogram of `data` and overlay a Gaussian (normal) PDF fit.

      This function computes the sample mean (μ) and sample standard deviation (σ, ddof=1),
      draws a density histogram, overlays the corresponding normal probability density
      function, and saves the figure as a JPEG named `histogram_<y_string>.jpg`.

      Parameters
      ----------
      data : array-like
        1D numeric samples to histogram (e.g., list, NumPy array, torch tensor converted
        to NumPy). NaNs/inf values should be removed beforehand.
      x_string : str
        Label for the x-axis (e.g., "TRs").
      y_string : str
        Label for the y-axis (e.g., "counts") and suffix used for the output filename.
      bins : int
        Number of histogram bins.
      counts_show: bool for selection counts in the plot or not

      Returns
      -------
      None
        The plot is saved to disk; nothing is returned.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if counts_show is True:
        counts, bin_edges, _ = ax.hist(data, bins=bins, density=False, alpha=0.6, edgecolor="black")
    else:
        ax.hist(data, bins=bins, density=True, alpha=0.6, edgecolor="black")

    mu = np.mean(data)
    sigma = np.std(data, ddof=1)

    xx = np.linspace(np.min(data), np.max(data), 500)
    normal_pdf = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xx - mu) / sigma) ** 2)

    if counts_show is True:
       # Scale pdf to match histogram counts
       bin_width = bin_edges[1] - bin_edges[0]
       normal_counts = normal_pdf * len(data) * bin_width
       ax.plot(xx, normal_counts, linewidth=2, label=f"Normal fit (μ={mu:.3g}, σ={sigma:.3g})")
    else:
       ax.plot(xx, normal_pdf, linewidth=2, label=f"Normal fit (μ={mu:.3g}, σ={sigma:.3g})")

    ax.grid()
    ax.set_xlabel(x_string)
    ax.set_ylabel(y_string)
    # ax.set_title(title)
    fig.tight_layout()

    if counts_show is True:
       fig.savefig(f"./histogram_{x_string}_counts.jpg")
    else:
       fig.savefig(f"./histogram_{x_string}.jpg")
    plt.close("all")

def collate_drop_none(batch):
    """
    Collate function that drops invalid samples returned as None by the Dataset.

    This is used to prevent DataLoader crashes when a subject is missing one or
       more modality files. If all samples in a batch are invalid, None is returned.

    Args:
       batch: List of dataset samples, where invalid samples may be None.

    Returns:
      A collated batch using PyTorch's default_collate with None entries removed,
      or None if the entire batch was invalid.
    """

    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # whole batch invalid
    return default_collate(batch)


# define the the Iterable Dataset for reading all the modalities at the
# same time


class StreamPairs(Dataset):
    """
    Multi-modal ENIGMA dataset that yields aligned RS, structural, and fALFF/ReHo data per subject.

    The dataset:
    - Uses a precomputed subject/site index structure (`subject_sites`) describing which
      subjects exist across RS and structural modalities per site.
    - Loads:
    * RSData: 4D parcellated NIfTI volumes (time dimension cropped/resampled)
    * Structural: precomputed NPZ tensors (surface/thickness/volume encodings)
    * fALFF/ReHo: precomputed NPZ tensors
    - Applies site-specific subject-ID normalization (e.g., case, underscores, suffix rules)
      to locate files consistently across heterogeneous ENIGMA naming conventions.
    - Optionally resamples temporal indices to a target sampling frequency and crops to a
      fixed window length for model input stability.

    Returned sample (conceptually):
       (index, rs_data, structural_data, falff_reho_data, subject_id, site_id)
    """

    def __init__(
        self,
        subject_sites,
        rs_path,
        st_path,
        falff_reho_path,
        rs_time_window=20,  # initial set of time_points used for -  each get_item define this in SECOND
        rs_window_crop=30,  # this should lower of equal to the final resample vector len
        rs_random_window=False,
        rs_stride=4,
        st_stride=4,
        falff_stride=3,
    ):
        """
        Initialize the dataset and (optionally) preload per-site NPZ containers.

        This constructor:
        - Stores paths for RS NIfTI directories and structural/fALFF-ReHo NPZ caches.
        - Reads the subject/site index arrays needed to iterate consistently across sites.
        - Defines special-case site rules for subject name formatting and file discovery.
        - Optionally loads per-site NPZ files into memory to reduce per-sample disk I/O.

        Args:
          subject_sites: Precomputed index structure describing overlapping subjects and sites.
          rs_path: Root folder for RSData per-site/per-subject NIfTI files.
          st_path: Folder containing structural NPZ caches (per site).
          falff_reho_path: Folder containing fALFF/ReHo NPZ caches (per site).
          rs_time_window: Maximum RS duration (seconds) to consider before cropping.
          rs_window_crop: Fixed number of timepoints to output after resampling/cropping.
        """

        # read here the TR_vals
        with open("../../../Data/npz/TR_vals.pkl", "rb") as file_TRs:
            self.tr_vals = pickle.load(file_TRs)

        # define here the initialization parameters
        self.subject_sites = subject_sites
        self.sub_rs = [str(item) for item in self.subject_sites["subjects_rs"]]
        self.sub_st = [str(item) for item in self.subject_sites["subjects_st"]]

        # define the sites for changing the subject suffixes
        self.special_sites_1 = ["Ghent", "Toledo", "Tours", "NanjingYixing", "Masaryk"]
        self.special_sites_2 = ["Beijing", "Capetown", "Cisler"]

        self.rs_path = rs_path
        self.st_path = st_path
        self.falff_reho_path = falff_reho_path
        self.rs_time_window = rs_time_window
        self.rs_random_window = rs_random_window
        self.rs_stride = rs_stride
        self.st_stride = st_stride
        self.falff_stride = falff_stride

        self.st_DATA = []
        self.rs_window_crop = rs_window_crop
        self.falff_reho_DATA = []
        self.st_subjects = []
        self.falff_subjects = []

        # get the unique list of sites
        self.sites = list(dict.fromkeys(self.subject_sites["sites_all"]))
        self.sites_all = self.subject_sites["sites_all"]

        if not os.path.exists(
            "../../../Data/npy/structural_npys_all.npy"
        ) and not os.path.exists("../../../Data/npy/falff_reho_npys_all.npy"):
            # read the files as list and save it as npys if necessary
            logger.info("Reading structural Data for all sites!!")

            # reading the whole sites data for structural
            for index_sites in range(0, len(self.sites)):
                self.st_data_site = np.load(
                    self.st_path
                    + "/structural_data_"
                    + self.sites[index_sites]
                    + "_data.npz",
                    mmap_mode="r",
                )

                # perform the decimation here
                self.st_DATA.append(
                    [
                        self.st_data_site["rs_tensor_3d_dest_surf"][
                            :, :: self.st_stride, :: self.st_stride, :: self.st_stride
                        ],
                        self.st_data_site["rs_tensor_3d_dest_thick"][
                            :, :: self.st_stride, :: self.st_stride, :: self.st_stride
                        ],
                        self.st_data_site["rs_tensor_3d_dest_vol"][
                            :, :: self.st_stride, :: self.st_stride, :: self.st_stride
                        ],
                    ]
                )
                self.st_subjects.append(self.st_data_site["subjects"][:, 1].tolist())
                logger.info(f"Read structural for site {self.sites[index_sites]}")

            # saving the st variables firt to do the del afterwards
            self.st_DATA = np.array(self.st_DATA, dtype=object)  # (N_total, X, Y, Z)
            np.save(
                "../../../Data/npy/structural_npys_all.npy",
                self.st_DATA,
                allow_pickle=True,
            )
            # write the pickle directory here
            with open("../../../Data/npy/sub_structural_pkl.pkl", "wb") as file_structural:
                pickle.dump(self.st_subjects, file_structural)

            # delete this temporary values to save RAM memory and keep speed
            del self.st_data_site, self.st_DATA

            # read the files as list and save it as npys if necessary
            logger.info("Reading falff_reho Data for all sites!!")

            # reading the whole sites data for falff reho
            for index_sites in range(0, len(self.sites)):
                self.falff_reho_data_site = np.load(
                    self.falff_reho_path
                    + "/falff_reho_data_"
                    + self.sites[index_sites]
                    + "_data.npz",
                    mmap_mode="r",
                )

                # save the reho data here
                self.falff_reho_DATA.append(
                    [
                        self.falff_reho_data_site["alff_tensor_3d"][
                            :,
                            :: self.falff_stride,
                            :: self.falff_stride,
                            :: self.falff_stride,
                        ],
                        self.falff_reho_data_site["falff_tensor_3d"][
                            :,
                            :: self.falff_stride,
                            :: self.falff_stride,
                            :: self.falff_stride,
                        ],
                        self.falff_reho_data_site["reho_tensor_3d"][
                            :,
                            :: self.falff_stride,
                            :: self.falff_stride,
                            :: self.falff_stride,
                        ],
                    ]
                )
                self.falff_subjects.append(
                    self.falff_reho_data_site["subjects"][:, 1].tolist()
                )
                logger.info(f"Read falff_reho for site {self.sites[index_sites]}")

            # save the interim files as npy for not reading them again using

            self.falff_reho_DATA = np.array(
                self.falff_reho_DATA, dtype=object
            )  # (N_total, X, Y, Z)
            np.save(
                "../../../Data/npy/falff_reho_npys_all.npy",
                self.falff_reho_DATA,
                allow_pickle=True,
            )

            # write the pickle directory here
            with open("../../../Data/npy/sub_falff_reho_pkl.pkl", "wb") as file_falff:
                pickle.dump(self.falff_subjects, file_falff)

            del self.falff_reho_data_site, self.falff_reho_DATA

        else:
            # define the values loading here having the data already defined. Let's
            # measure how it takes reading the files
            logger.info("Reading pre-saved dataset!!")
            self.falff_reho_DATA = np.load(
                "../../../Data/npy/falff_reho_npys_all.npy", allow_pickle=True
            )
            self.st_DATA = np.load(
                "../../../Data/npy/structural_npys_all.npy", allow_pickle=True
            )

            # read the subjects here using pickle load
            with open("../../../Data/npy/sub_falff_reho_pkl.pkl", "rb") as file_alff:
                self.falff_subjects = pickle.load(file_alff)

            # read the subjects here using pickle load
            with open("../../../Data/npy/sub_structural_pkl.pkl", "rb") as file_structural:
                self.st_subjects = pickle.load(file_structural)

            logger.info("Reading finalized!!")

    def __len__(self):
        """
        Return the total number of subject samples available in the index structure.
        Returns:
         Number of samples the dataset can yield (subject-major across all sites)
        """
        return len(self.sub_rs)

    def __getitem__(self, idx):
        """
        Load and return a single multimodal sample for a given dataset index.

        The getter:
        - Resolves site + subject identifiers for the requested index.
        - Applies site-specific normalization to locate RS NIfTI files reliably.
        - Loads RS 4D volume(s), then resamples/crops the time dimension to a fixed
          window using integer-safe index selection.
        - Loads structural and fALFF/ReHo arrays from per-site NPZ caches and selects
          the matching subject entry.
        - Returns None when required files are missing or data integrity checks fail.

        Args:
          idx: Dataset index.

        Returns:
           A tuple containing:
              - idx: original dataset index
              - rs_data: cropped/resampled RS tensor 4D (model-ready)
              - st_data_input: structural tensor 3D (model-ready)
              - falff_reho_data_input: fALFF/ReHo tensor 3D (model-ready)
              - subject_id: subject identifier string
              - site_id: site identifier string
          Or:
              None if the sample is invalid (missing files, mismatched IDs, or failed checks).
        """

        rs_data_path_1 = (
            self.rs_path
            + "/"
            + self.sites_all[idx]
            + "/"
            + self.subject_sites["subjects_rs"][idx]
            + "/"
            + self.subject_sites["subjects_rs"][idx]
            + "_brainnetome_4d_mni_image.nii.gz"
        )
        rs_data_path_2 = (
            self.rs_path
            + "/"
            + self.sites_all[idx]
            + "/"
            + self.subject_sites["subjects_st"][idx]
            + "/"
            + self.subject_sites["subjects_st"][idx]
            + "_brainnetome_4d_mni_image.nii.gz"
        )

        # do the validation in the first part
        index_site = self.sites.index(self.sites_all[idx])
        # replacing the suffixes of st_subjects here before comparing
        if self.sites_all[idx] == "UMN":
            self.st_subjects[index_site] = [
                s.replace("_", "") for s in self.st_subjects[index_site]
            ]
        elif self.sites_all[idx] == "Milwaukee":
            self.st_subjects[index_site] = [
                s.replace("6mo_", "") for s in self.st_subjects[index_site]
            ]
        elif self.sites_all[idx] == "UWash":
            self.st_subjects[index_site] = [
                s.replace("R", "") for s in self.st_subjects[index_site]
            ]
        elif self.sites_all[idx] == "Leiden":
            self.st_subjects[index_site] = [
                s.replace("Episca", "S") for s in self.st_subjects[index_site]
            ]

        # do this validation as first to avoid problems with other sites
        if self.sub_rs[idx] in self.st_subjects[index_site]:
            subject_index_st = self.st_subjects[index_site].index(
                self.subject_sites["subjects_rs"][idx]
            )
        if self.sub_st[idx] in self.st_subjects[index_site]:
            subject_index_st = self.st_subjects[index_site].index(
                self.subject_sites["subjects_st"][idx]
            )
        if self.sub_rs[idx] in self.falff_subjects[index_site]:
            subject_index_rs = self.falff_subjects[index_site].index(
                self.subject_sites["subjects_rs"][idx]
            )
        if self.sub_st[idx] in self.falff_subjects[index_site]:
            subject_index_rs = self.falff_subjects[index_site].index(
                self.subject_sites["subjects_st"][idx]
            )

        if (
            self.sites_all[idx] not in self.special_sites_1
            and self.sites_all[idx] not in self.special_sites_2
        ):
            subject_rs_path_1 = self.subject_sites["subjects_rs"][idx]
            subject_rs_path_2 = self.subject_sites["subjects_st"][idx]
            rs_data_path_1 = (
                self.rs_path
                + "/"
                + self.sites_all[idx]
                + "/"
                + subject_rs_path_1
                + "/"
                + subject_rs_path_1
                + "_brainnetome_4d_mni_image.nii.gz"
            )
            rs_data_path_2 = (
                self.rs_path
                + "/"
                + self.sites_all[idx]
                + "/"
                + subject_rs_path_2
                + "/"
                + subject_rs_path_2
                + "_brainnetome_4d_mni_image.nii.gz"
            )
        elif self.sites_all[idx] in self.special_sites_1:
            subject_rs_path_1 = self.falff_subjects[index_site][subject_index_rs]
            subject_rs_path_2 = self.subject_sites["subjects_st"][idx]
            if self.sites_all[idx] == "Toledo":
                rs_data_path_1 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1
                    + "/"
                    + subject_rs_path_1.replace("M", "m").replace("O", "o")
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
            elif self.sites_all[idx] == "NanjingYixing":
                rs_data_path_1 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1
                    + "/"
                    + subject_rs_path_1.replace("S", "s")
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
            else:
                rs_data_path_1 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1
                    + "/"
                    + subject_rs_path_1.lower()
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
            rs_data_path_2 = (
                self.rs_path
                + "/"
                + self.sites_all[idx]
                + "/"
                + subject_rs_path_2
                + "/"
                + subject_rs_path_2.lower()
                + "_brainnetome_4d_mni_image.nii.gz"
            )
        elif self.sites_all[idx] in self.special_sites_2:
            # validates if subject_index_st exists
            if "subject_index_st" in locals():
                subject_rs_path_1 = self.st_subjects[index_site][subject_index_st]
            else:
                subject_rs_path_1 = self.subject_sites["subjects_rs"][idx]

            if self.sites_all[idx] == "Cisler":
                rs_data_path_1 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1.replace("DOP", "DOP_").replace("PAL", "PAL_")
                    + "/"
                    + subject_rs_path_1.replace("DO", "dO").replace("PA", "pA")
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
            elif self.sites_all[idx] == "Capetown":
                rs_data_path_11 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1.replace("sub-", "sub-capetown-")
                    + "/"
                    + subject_rs_path_1
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
                rs_data_path_12 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1.replace("sub-", "sub-tygerberg-")
                    + "/"
                    + subject_rs_path_1
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
                if os.path.exists(rs_data_path_11):
                    rs_data_path_1 = rs_data_path_11
                if os.path.exists(rs_data_path_12):
                    rs_data_path_1 = rs_data_path_12
            elif self.sites_all[idx] == "Beijing":
                rs_data_path_1 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1[0:4]
                    + f"{int(subject_rs_path_1[4:]) + 880:03d}"
                    + "/"
                    + subject_rs_path_1[0:4]
                    + f"{int(subject_rs_path_1[4:]) + 880:03d}"
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
            else:
                rs_data_path_1 = (
                    self.rs_path
                    + "/"
                    + self.sites_all[idx]
                    + "/"
                    + subject_rs_path_1
                    + "/"
                    + subject_rs_path_1
                    + "_brainnetome_4d_mni_image.nii.gz"
                )
            subject_rs_path_2 = self.subject_sites["subjects_st"][idx]
            rs_data_path_2 = (
                self.rs_path
                + "/"
                + self.sites_all[idx]
                + "/"
                + subject_rs_path_2
                + "/"
                + subject_rs_path_2
                + "_brainnetome_4d_mni_image.nii.gz"
            )

        # evaluate the existence of rs_data_path
        if os.path.exists(rs_data_path_1):
            # read here the RSData
            rs_img = nib.load(rs_data_path_1, mmap=True)

        elif os.path.exists(rs_data_path_2):
            # read here the RSData
            rs_img = nib.load(rs_data_path_2, mmap=True)
        else:
            logger.error(f"RSData does not exist for path {rs_data_path_1}")
            return None

        logger.info(
            f"Read RSdata from site {self.sites_all[idx]} subject {self.subject_sites['subjects_rs'][idx]} with RSData path {rs_data_path_1} or {rs_data_path_2}"
        )

        # resample the index vector first
        fs_current = 1 / self.tr_vals[self.sites_all[idx]]
        fs_new = 1 / self.tr_vals["max"]

        # validate maximum length here - to avoud problems with the maximum time-series length
        if int(self.rs_time_window * fs_current) <= rs_img.dataobj.shape[3]:
            idxs_vals = np.arange(0, int(self.rs_time_window * fs_current), 1)
        else:
            idxs_vals = np.arange(0, rs_img.dataobj.shape[3], 1)

        # take the length of each subject for plotting histogram lengths
        time_length = float(rs_img.dataobj.shape[3] / fs_current)

        if time_length <= 200:
           # print this in the report to check the ids for the shorter trials
           logger.error(f"The data length is very short and its length is {time_length} secs for subject {self.subject_sites['subjects_rs'][idx]} and site {self.sites_all[idx]}")
           # skip this subject because it is too short to take enough time information
           continue

        # do the resample using nearest approach
        new_in_current = np.round(idxs_vals * fs_current / fs_new).astype(int)

        # do this in traditional way for integer labels
        idx_resamp = np.rint(new_in_current).astype(int)
        idx_resamp = np.clip(idx_resamp, 0, len(idxs_vals) - 1)
        # remove the duplicates out of the batch and clip it to the maximum value
        new_index = idxs_vals[idx_resamp]
        new_index[0 : self.rs_window_crop] = np.clip(
            new_index[0 : self.rs_window_crop], 0, len(idxs_vals) - 1
        )

        # do the indexing one by one to avoid memory overloading and faster processing
        rs_data = np.stack(
            [
                np.asarray(
                    rs_img.dataobj[
                        :: self.rs_stride, :: self.rs_stride, :: self.rs_stride, int(t)
                    ]
                )
                for t in new_index[0 : self.rs_window_crop]
            ],
            axis=-1,
        )

        # older version*** UNCOMMENT IT IF CONSIDER**
        # rs_data = np.asarray(
        # rs_img.dataobj[::self.rs_stride, ::self.rs_stride, ::self.rs_stride,
        # :])[..., new_index[0:self.rs_window_crop]]

        logger.info(
            f"Current TR is {self.tr_vals[self.sites_all[idx]]} and target TR is {self.tr_vals['max']}.The new 4D array resampled has {len(new_index[0 : self.rs_window_crop])} samples in time and original resample is {len(new_index)} with samples {new_index[0 : self.rs_window_crop]}, the idx of the original time-series is {idxs_vals}, time_length per subject is {time_length}"
        )

        # replacing the suffixes of st_subjects here before comparing
        if self.sites_all[idx] == "UMN":
            self.st_subjects[index_site] = [
                s.replace("_", "") for s in self.st_subjects[index_site]
            ]
        elif self.sites_all[idx] == "Milwaukee":
            self.st_subjects[index_site] = [
                s.replace("6mo_", "") for s in self.st_subjects[index_site]
            ]
        elif self.sites_all[idx] == "UWash":
            self.st_subjects[index_site] = [
                s.replace("R", "") for s in self.st_subjects[index_site]
            ]
        elif self.sites_all[idx] == "Leiden":
            self.st_subjects[index_site] = [
                s.replace("Episca", "S") for s in self.st_subjects[index_site]
            ]
        elif self.sites_all[idx] == "Cisler":
            self.st_subjects[index_site] = [
                s.replace("_", "S") for s in self.st_subjects[index_site]
            ]

        if self.sub_rs[idx] in self.st_subjects[index_site]:
            subject_index_st = self.st_subjects[index_site].index(
                self.subject_sites["subjects_rs"][idx]
            )
        if self.sub_st[idx] in self.st_subjects[index_site]:
            subject_index_st = self.st_subjects[index_site].index(
                self.subject_sites["subjects_st"][idx]
            )
        if self.sub_rs[idx] in self.falff_subjects[index_site]:
            subject_index_rs = self.falff_subjects[index_site].index(
                self.subject_sites["subjects_rs"][idx]
            )
        if self.sub_st[idx] in self.falff_subjects[index_site]:
            subject_index_rs = self.falff_subjects[index_site].index(
                self.subject_sites["subjects_st"][idx]
            )

        if "subject_index_st" not in locals():
            return None

        # get the site value for st and reho
        st_data_input = [
            self.st_DATA[index_site][0][subject_index_st, :],
            self.st_DATA[index_site][1][subject_index_st, :],
            self.st_DATA[index_site][2][subject_index_st, :],
        ]
        falff_reho_data_input = [
            self.falff_reho_DATA[index_site][0][subject_index_rs, :],
            self.falff_reho_DATA[index_site][1][subject_index_rs, :],
            self.falff_reho_DATA[index_site][2][subject_index_rs, :],
        ]

        logger.info(
            f"RS index {subject_index_rs} and ST index {subject_index_st}, for subject {self.falff_subjects[index_site][subject_index_rs]} in rs, and subject {self.st_subjects[index_site][subject_index_st]} in st"
        )

        # return the tuple with the values corresponding with the same subject
        # information
        return (
            idx,
            rs_data,
            st_data_input,
            falff_reho_data_input,
            self.subject_sites["subjects_rs"][idx],
            self.sites_all[idx],
            new_index[0 : self.rs_window_crop],
            time_length,
            self.tr_vals[self.sites_all[idx]]
        )


def define_dataset_dataloader_ENIGMA(subject_indices_current_data: str):
    """
    Construct the ENIGMA StreamPairs dataset and a PyTorch DataLoader.

    This function:
     - Loads a saved NumPy structure containing subject/site indices.
     - Instantiates `StreamPairs` with configured paths and RS window parameters.
     - Builds a DataLoader using a robust collate function that drops invalid samples.

     Args:
       subject_indices_current_data: (str) Path to a .npy file containing the subject/site index object.

     Returns:
       data_loader_all: A PyTorch DataLoader object yielding multimodal batches suitable for training/eval. For the SSL section ** For now
    """

    # define the dataset here and invoke the initialization call
    subject_sites = np.load(subject_indices_current_data)
    dataset_all = StreamPairs(
        subject_sites=subject_sites,
        rs_path=RSdata_folder,
        st_path=structural_npz_folder,
        falff_reho_path=falff_reho_npz_folder,
        rs_time_window=200,
        rs_window_crop=56,
    )  # don't increase rs_window_crop more than 56***

    current_batch_size = 10

    # define the Dataloder here
    data_loader_all = DataLoader(
        dataset_all,
        batch_size=current_batch_size,
        shuffle=True,
        collate_fn=collate_drop_none,  # define this to skip the unexistent data
        pin_memory=True,
    )

    # return the dataloader here
    return data_loader_all


"""
  **Main section of the code**
"""

# read the dataloder object here
data_loader_all = define_dataset_dataloader_ENIGMA(
    subject_indices_current_data=subject_indices_current_data
)

start_time = time.time()

# check if the dataloader works
idx_sample = []
sites_sample = []
tr_values = []
time_sub_values = []
for batch_data in data_loader_all:
    if batch_data is None:  # validate this when batch is None and skip
        continue

    idx, rs_DATA, st_DATA, falff_reho_DATA, subject_index, sites_idx, sampling_index, time_subject, TRs = (
        batch_data
    )
    tr_values.append(TRs)
    idx_sample.append(idx)
    time_sub_values.append(time_subject)
    sites_sample.append(sites_idx)
    logger.info(f"Reading modalities for subject {subject_index}")

time_sub_values = [str(s) for sublist in time_sub_values for s in sublist]
time_sub_values_np = np.asarray(time_sub_values)
tr_values = [str(s) for sublist in tr_values for s in sublist]
tr_values = np.array([float(re.search(r"[-+]?\d*\.?\d+", s).group()) for s in tr_values],dtype=np.float32)


time_sub_values_np = np.array([float(re.search(r"[-+]?\d*\.?\d+", s).group()) for s in time_sub_values_np],dtype=np.float64)
sites_flat = [str(s) for sublist in sites_sample for s in sublist]
sites_unique = list(dict.fromkeys(sites_flat))

# check the final time of dataloder reader
end_time = time.time()

logger.info(f"dataloader final time reading after decimation {end_time - start_time} s")
logger.info(f"subjects with all modalities {len(torch.cat(idx_sample))}")
logger.info(
    f"data from sites {sites_unique} are taking into account in this dataloader"
)

# convert the report here defined in the loguru configuration above coloured
conv = Ansi2HTMLConverter(inline=True)
ansi = open("report_final_excluded_subjects.log", encoding="utf-8").read()
html = f"""<!doctype html><meta charset="utf-8">
<body style="background:#111;color:#eee;font-family:monospace;white-space:pre-wrap">
{conv.convert(ansi, full=False)}
</body>"""

open("report_final_excluded_subjects.html", "w", encoding="utf-8").write(html)

## do the histogram plot here
plot_histogram(data=time_sub_values_np, x_string="trial length [s]", y_string="probability", bins=10, counts_show=False)
plot_histogram(data=tr_values, x_string="TR [s]", y_string="probability", bins=5, counts_show=False)
plot_histogram(data=time_sub_values_np, x_string="trial length [s]", y_string="counts", bins=10, counts_show=True)
plot_histogram(data=tr_values, x_string="TR [s]", y_string="counts", bins=5, counts_show=True)
