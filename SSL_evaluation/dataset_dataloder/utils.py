"""
   dataloader utils module
   for using in the dataloader and dataset
   definition
"""
import pandas as pd
from loguru import logger
 
def reading_ENIGMA_excel_sheet(df_ENIGMA, site: str, subject: str, verbose: bool):
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

    Returns:
      A pandas DataFrame (or equivalent structured object) containing normalized
      subject identifiers for that site, suitable for overlap matching.

    Notes:
      - The function assumes a fixed sheet layout (skiprows/usecols) used by ENIGMA.
      - Identifier normalization is performed to reduce site-specific naming issues.
    """

    if site == "Muenster":
        site_search = ["munster"]
    elif site == "MinnVA":
        site_search = ["minn_va"]
    elif site == "Cisler":
        site_search = ["uw_cisler"]
    elif site == "Lawson":
        site_search = ["ontario"]
    elif site == "WacoVA":
        site_search = ["waco_va"]
    elif site == "Grupe":
        site_search = ["uw_grupe"]
    elif site == "Capetown":
        site_search = ["capetown_capetown", "capetown_tygerberg"]
    elif site == "McLean":
        site_search = ["mclean_rosso", "mclean_kaufman"]
    elif site == "NanjingYixing":
        site_search = ["nanjing"]
    else:
        site_search = [site.lower()]


    subject_search = str(subject).replace("sub-","").replace("sub","")
    if site == "UWash":
         subject_search = "R" + subject_search
    elif site == "Beijing":
         subject_search = "sub_" + subject_search

    # set headers from row 0
    df_ENIGMA.columns = df_ENIGMA.iloc[0].astype(str).str.strip()

    # remove that header row from the data
    df_ENIGMA = df_ENIGMA.iloc[1:].reset_index(drop=True)

    df_site = df_ENIGMA[
        df_ENIGMA["site"].astype(str).str.strip().str.lower().isin(
            [s.strip().lower() for s in site_search]
        )
    ]

    # take the subject in the second column referring to the unified subject in the dataloader
    # definition in the upper loop
    if site != "Ghent" and site != "Tours":
       df_subject = df_site[df_site["SubjID2"].astype(str).str.strip() == subject_search]
    else:
       df_subject = df_site[df_site["SubjID"].astype(str).str.strip() == site + "_" + subject_search]

    if df_subject.empty:
       df_subject = df_site[df_site["ID"].astype(str).str.strip() == subject_search]
       if df_subject.empty:
           df_subject = df_site[df_site["ID"].astype(str).str.strip() == subject_search + "_doNotUse"]
           # some subject will be empty in any case we need to validate them out
           if df_subject.empty:
              if verbose is True:
                 logger.info("Data for {str(subject)} and site {site} is not existing the main spreadsheet! returning empty")
              df_subject = None
    else:
       if verbose is True:
          logger.info(f"Data from subject {str(subject)} and site {site} in the ENIGMA main spreadsheet is:")
          print(df_subject)

    return df_subject
