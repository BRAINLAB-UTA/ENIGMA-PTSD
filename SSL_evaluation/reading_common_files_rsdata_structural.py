"""
This modules contains the utilities and a driver script to align RSData subjects with structural/fALFF-ReHo subjects per site.

This module provides:
- A helper to read and normalize subject identifiers from ENIGMA structural spreadsheets.
- A top-level workflow that loads per-site NPZ datasets (structural + fALFF/ReHo),
  compares subject lists against RSData folders, and builds aligned subject/site lists
  for downstream multi-modal training or evaluation.

Typical use:
  Run once to generate unified lists of overlapping subjects across modalities.
  Take in mind the right path for the *.csvs with the structural path
"""

import os
import warnings

import numpy as np
import pandas as pd
from ansi2html import Ansi2HTMLConverter
from loguru import logger

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

logger.add("report.log", format=FMT, colorize=True)

# getting the auxiliary function for reading the corresponding IDs from the structural


def reading_excel_sheet_corresponding_ids(path_file: str, equivalent_site: str):
    """
    Extract and normalize subject identifiers from an ENIGMA structural Excel sheet.

    The structural spreadsheets often contain multiple ID columns and site-specific
    formatting. This helper reads the first three columns of the sheet (rows below
    the header block), filters to the requested site, and returns a cleaned mapping
    used to reconcile subject naming differences between:
      - structural spreadsheets
      - RSData folder names
      - derived numpy containers

    Args:
      path_file: Path to the structural Excel file for a given ENIGMA site group.
      equivalent_site: Site name/key used to select the appropriate entries.

    Returns:
      A pandas DataFrame (or equivalent structured object) containing normalized
      subject identifiers for that site, suitable for overlap matching.

    Notes:
      - The function assumes a fixed sheet layout (skiprows/usecols) used by ENIGMA.
      - Identifier normalization is performed to reduce site-specific naming issues.
    """

    df_site = pd.read_excel(
        path_file,
        sheet_name=0,
        usecols=[0, 1, 2],  # first 3 columns (A:C)
        skiprows=4,  # start reading at row 5 (0-based skip)
        header=None,  # no header in those rows
        names=["site", "id", "subject"],
    )

    # kflattened_subjects_rseep only rows that look like your mapping
    df_subjects_overlapped = df_site[
        df_site["site"].astype(str).str.lower().eq(equivalent_site)
    ].reset_index(drop=True)

    # read the corresponding subject on each of the ids given there for
    # matching with other modalities
    st_subjects = df_subjects_overlapped["id"].tolist()
    rs_subjects_col = df_subjects_overlapped["subject"].tolist()

    rs_subjects_col = [
        s.replace(f"{equivalent_site.upper()}_", "") for s in rs_subjects_col
    ]

    if equivalent_site == "uwash":
        st_subjects = [s.replace("R", "") for s in st_subjects]
        rs_subjects_col = [s.replace("UWash_", "") for s in rs_subjects_col]
    elif equivalent_site == "capetown":
        rs_subjects_col = [
            s.replace("Capetown_", "")
            .replace("capetown_", "")
            .replace("tygerberg_", "")
            for s in rs_subjects_col
        ]
    elif equivalent_site == "ghent":
        rs_subjects_col = [s.replace("Ghent_", "") for s in rs_subjects_col]
    elif equivalent_site == "tours":
        rs_subjects_col = [s.replace("Tours_", "") for s in rs_subjects_col]
    elif equivalent_site == "vanderbilt":
        rs_subjects_col = [s.replace("Vanderbilt_", "") for s in rs_subjects_col]
    elif equivalent_site == "groningen":
        rs_subjects_col = [s.replace("Groningen_", "") for s in rs_subjects_col]
    elif equivalent_site == "beijing":
        rs_subjects_col = [s.replace("Beijing_", "") for s in rs_subjects_col]
    elif equivalent_site == "munster":
        rs_subjects_col = [s.replace("Muenster_", "") for s in rs_subjects_col]
    elif equivalent_site == "minn_va":
        rs_subjects_col = [s.replace("MinnVA_", "") for s in rs_subjects_col]
    elif equivalent_site == "uw_cisler":
        rs_subjects_col = [
            s.replace("Cisler_", "").replace("_", "") for s in rs_subjects_col
        ]
        st_subjects = [s.replace("_", "") for s in st_subjects]
    elif equivalent_site == "leiden":
        rs_subjects_col = [s.replace("Leiden_", "") for s in rs_subjects_col]
        st_subjects = [s.replace("Episca", "S") for s in st_subjects]

    rs_subjects_col = [f"sub-{s}" for s in rs_subjects_col]
    st_subjects = [f"sub-{s}" for s in st_subjects]

    # returning the equivalent pair in the structural excel sheet
    return st_subjects, rs_subjects_col


"""
   *** Main section of the code here***
"""

# define the path suffixes here
RSdata_folder = "../../Data/RSData/"
npz_folder = "../../Data/npz"
falff_FOLDER = "../../Data/falff_reho/"

structural_npz_folder = f"{npz_folder}/structural"
falff_reho_npz_folder = f"{npz_folder}/falff_reho_3d"

path_to_structural_spreadsheet = (
    "../../Data/Structural/CorticalMeasuresENIGMA_DKT40_ThickAvg.xlsx"
)

DATA_structural = []
DATA_falff_reho = []
SITES = []
SITES_ALL = []
SUB_RS = []
SUB_ST = []

review_sites = [
    "AMC",
    "Beijing",
    "Capetown",
    "Ghent",
    "Groningen",
    "MinnVA",
    "Leiden",
    "Muenster",
    "Tours",
    "UWash",
    "Vanderbilt",
    "Cisler",
]

# then move around the RSData directory and read both modalities to
# compare subjects and select the ones overlapped
for site in os.listdir(RSdata_folder):
    subjects_rs = []
    subjects_st = []
    sites_interim = []
    # read here the site name and compare for reading the site from inside
    if "." not in site and site != "Utrecht" and site != "Columbia":
        structural_npz_FOLDER = (
            f"{structural_npz_folder}/structural_data_{site}_data.npz"
        )
        falff_reho_FOLDER = f"{falff_reho_npz_folder}/falff_reho_data_{site}_data.npz"
        SData = np.load(structural_npz_FOLDER)
        fALFFData = np.load(falff_reho_FOLDER)

        if site == "Cisler":
            rs_data = "uw_cisler"
        elif site == "Grupe":
            rs_data = "uw_grupe"
        elif site == "Lawson":
            rs_data = "ontario"
        elif site == "McLean":
            rs_data = "mclean_kaufman"
        elif site == "MinnVA":
            rs_data = "minn_va"
        elif site == "Muenster":
            rs_data = "munster"
        elif site == "NanjingYixing":
            rs_data = "nanjing"
        elif site == "WacoVA":
            rs_data = "waco_va"
        else:
            rs_data = site.lower()

        SITES.append(site)
        # validate if the site is inside in any of the valid sites for revalidation
        if site in review_sites:
            sub_st_val, sub_rs_val = reading_excel_sheet_corresponding_ids(
                path_file=path_to_structural_spreadsheet, equivalent_site=rs_data
            )
        else:
            sub_st_val = []
            sub_rs_val = []

        # analyze here the list in case this is necessary
        if len(SData["subjects"]) >= len(fALFFData["subjects"]):
            base_subject_list = [x[1] for x in fALFFData["subjects"]]
            target_subject_list = [x[1] for x in SData["subjects"]]
            falff_indicator = True
        else:
            base_subject_list = [x[1] for x in SData["subjects"]]
            target_subject_list = [x[1] for x in fALFFData["subjects"]]
            falff_indicator = False

        base_subject_list = [str(x) for x in base_subject_list]
        target_subject_list = [str(x) for x in target_subject_list]

        logger.info(
            f"There are {len(SData['subjects'])} subjects in the structural data and {
                len(fALFFData['subjects'])
            } subjects in the fALFF_reho for site {site}"
        )

        # take into account the index and the data between the lists
        for index_subj in range(0, len(base_subject_list)):
            # do initial validation here
            if site == "UMN":
                target_subject_list[index_subj] = target_subject_list[
                    index_subj
                ].replace("_", "")
                # breakpoint()
            elif site == "Milwaukee":
                target_subject_list[index_subj] = target_subject_list[
                    index_subj
                ].replace("6mo_", "")
            elif site == "UWash":
                target_subject_list[index_subj] = target_subject_list[
                    index_subj
                ].replace("R", "")
            elif site == "Leiden":
                base_subject_list[index_subj] = base_subject_list[index_subj].replace(
                    "Episca", "S"
                )
            elif site == "Cisler":
                target_subject_list[index_subj] = target_subject_list[
                    index_subj
                ].replace("_", "")

            # take into account what subjects are getting
            if (target_subject_list[index_subj] in base_subject_list) or (
                target_subject_list[index_subj] in sub_st_val
            ):
                if site in review_sites:
                    if target_subject_list[index_subj] in sub_st_val:
                        index_interim = sub_st_val.index(
                            target_subject_list[index_subj]
                        )
                    else:
                        logger.error(
                            f"subject {
                                target_subject_list[index_subj]
                            } is not in structural for site {site}"
                        )
                        continue
                    if sub_rs_val[index_interim] in base_subject_list:
                        index_data = base_subject_list.index(sub_rs_val[index_interim])
                    else:
                        logger.error(
                            f"subject {
                                target_subject_list[index_subj]
                            } is not in structural for site {site}"
                        )
                        continue
                else:
                    index_data = base_subject_list.index(
                        target_subject_list[index_subj]
                    )
                logger.success(
                    f"subject {
                        target_subject_list[index_subj]
                    } exist in all modalities for site {site}"
                )
            else:
                logger.error(
                    f"subject {
                        target_subject_list[index_subj]
                    } is not in structural for site {site}"
                )
                continue

            if site == "Leiden":
                subjects_rs.append(target_subject_list[index_subj])
                subjects_st.append(base_subject_list[index_data])
            else:
                subjects_st.append(target_subject_list[index_subj])
                subjects_rs.append(base_subject_list[index_data])

            sites_interim.append(site)

    # save the values per site here in a new npz file
    SUB_RS.append(subjects_rs)
    SUB_ST.append(subjects_st)
    SITES_ALL.append(sites_interim)

# get here the flattened list of subjects here
flattened_subjects_rs = [x for sublist in SUB_RS for x in sublist]
flattened_subjects_st = [x for sublist in SUB_ST for x in sublist]
flattened_sites = [x for sublist in SITES_ALL for x in sublist]
logger.success(
    f"The total amount of subject sharing RS, ST, and falff_reho modalities for sites {
        SITES
    } is {len(flattened_subjects_rs)}!"
)

# just do this as a validation for what subjects has been find as overlap
print(flattened_subjects_rs)
print(flattened_subjects_st)
print(flattened_subjects_rs == flattened_subjects_st)

# saving here the resulting files
if not os.path.exists("../../Data/npz/subjects_overlaped_all_modalities.npz"):
    logger.info("Saving the subjects overlap file in the Data directory...")

    np.savez_compressed(
        "../../Data/npz/subjects_overlaped_all_modalities.npz",
        subjects_rs=np.array(flattened_subjects_rs, dtype=str),
        subjects_st=np.array(flattened_subjects_st, dtype=str),
        sites_all=np.array(flattened_sites, dtype=str),
    )


if flattened_subjects_rs != flattened_subjects_st:
    for i in range(len(flattened_subjects_rs)):
        if flattened_subjects_rs[i] != flattened_subjects_st[i]:
            logger.info(
                f"File exist as '{flattened_subjects_rs[i]}' in rs_files and as '{
                    flattened_subjects_st[i]
                }' in st_files for site {flattened_sites[i]}"
            )

# Convert to HTML
conv = Ansi2HTMLConverter(inline=True)
ansi = open("report.log", encoding="utf-8").read()
html = f"""<!doctype html><meta charset="utf-8">
<body style="background:#111;color:#eee;font-family:monospace;white-space:pre-wrap">
{conv.convert(ansi, full=False)}
</body>"""

open("report.html", "w", encoding="utf-8").write(html)
