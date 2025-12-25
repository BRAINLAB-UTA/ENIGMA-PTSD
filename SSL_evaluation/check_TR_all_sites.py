"""
Scan ENIGMA RSData folders and summarize sampling frequency (TR) per site.

This script traverses the ENIGMA resting-state directory structure, reads each
site’s first available JSON sidecar (expected to contain "SamplingFrequency"),
and stores a per-site mapping plus the maximum observed value.

Outputs:
  - A dictionary saved as a pickle with per-site sampling frequency and a "max" entry.

Notes:
  - Although referred to as "TR" in comments, the script reads "SamplingFrequency".
  - The script stops after the first subject/file found per site for speed.
"""

import glob
import json
import os
import pickle

import numpy as np
from loguru import logger

"""
  This simple module
  checks all the TRs across
  all the sites in ENIGMA
"""

path_rs_data = "../../Data/RSData/"

TR_vals = {}
values_TR = []
for site in os.listdir(path_rs_data):
    #  continue
    if "." not in site:
        # check one subject and site
        for subjects in os.listdir(path_rs_data + "/" + site):
            # read here the information necessary to read the folder path
            pattern = os.path.join("/".join([path_rs_data, site, subjects]), "*.json")
            json_files = glob.glob(pattern)

            for json_file in json_files:
                # do the things here
                with open(json_file) as file:
                    exp_data = json.load(file)

                # This is TR not fs
                value_fs = exp_data["SamplingFrequency"]
                TR_vals[site] = value_fs
                values_TR.append(value_fs)
                logger.info(f"for site {site} the Sampling frequency is {value_fs}")
                break
            break

TR_vals["max"] = np.max(np.asarray(values_TR))

# Saving (writing in binary mode 'wb')
with open("../../Data/npz/TR_vals.pkl", "wb") as file:
    pickle.dump(TR_vals, file, pickle.HIGHEST_PROTOCOL)
