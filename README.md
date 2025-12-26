<table align="center" border="0" cellpadding="0" cellspacing="0" style="border: none; border-collapse: collapse;">
  <tr>
    <td align="center" style="border: none; padding-right:24px;">
      <a href="https://brainlab-uta.github.io/BRAIN-Lab/team.html" target="_blank" rel="noopener noreferrer">
        <img src="assets/uta_brain_lab.png" alt="BRAIN Lab @ UTA" height="70">
      </a>
    </td>
    <td align="center" style="border: none; padding-right:24px;">
      <a href="https://enigma.ini.usc.edu/ongoing/enigma-ptsd-working-group/" target="_blank" rel="noopener noreferrer">
        <img src="assets/enigma_name.png" alt="ENIGMA Consortium" height="100">
      </a>
    </td>
    <td align="center" style="border: none; padding-right:24px;">
      <a href="https://enigma.ini.usc.edu/" target="_blank" rel="noopener noreferrer">
        <img src="assets/enigma_logo.png" alt="ENIGMA Logo" height="200">
      </a>
    </td>
  </tr>
</table>

<h1 align="center">ENIGMA-PTSD</h1>
<h2  align="center"> 3D/4D parcellated image generation </h2>
  
<p align="center">
   Codes for pre-processing, dataloader creating, and Self-Supervised Learning evaluation over current ENIGMA-PTSD dataset. For accessing ENIGMA-PTSD data please contact contact <a href="mailto:xi.zhu@uta.edu" target="_blank" rel="noopener noreferrer"> <b>Dr. Xi Zhu</b></a>.
</p>

This repository contains:
- **Preprocessing pipelines** to generate in this [folder](https://github.com/BRAINLAB-UTA/ENIGMA-PTSD/tree/main/preprocessing):
  - structural features (sMRI-derived)
  - fALFF/ReHo (rs-fMRI-derived)
  - RSData (resting-state 4D time-series–derived)
- **Dataloader creation utilities** for downstream SSL evaluation/training in this [folder](https://github.com/BRAINLAB-UTA/ENIGMA-PTSD/tree/main/SSL_evaluation).

For start running the preprocessing and DataLoader creation follow the next steps in sequence:

## Quick start (environment)

### 1) Clone
```bash
git clone https://github.com/BRAINLAB-UTA/ENIGMA-PTSD.git
cd ENIGMA-PTSD
```

### 2) Create Python environment (Python version >= 3.11)

Please install **pip** before anything

Using **Conda**

```bash
conda create -n enigma-ptsd python=3.10 -y
conda activate enigma-ptsd
pip install -r requirements.txt
```

Using **venv**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 3) Data generation (structural, fALFF/ReHo, RSData)

Here we describe the intended flow for producing the derived files needed by the dataloader.

You generally need (per subject) on each modality folder:
   - T1w anatomical image 3D NIfTI .nii.gz in the folder **Structural**
   - Resting-state fMRI (4D NIfTI) + metadata TR, etc written in .json files in the folder **RSData**
   - ALFF/fALFF/ReHO 3D images in the folder **falffReHo**
   - Site/subject mapping tables (IDs, site names), as used by the ENIGMA project in main **ENIGMA anotation spreadsheet**

The structure of each folder modality will be like this:

```text
/DATA_Modality/
  siteA/
    sub-XXXX/
        T1w.nii.gz
        rest.nii.gz
        falff_reho.nii.gz
  siteB/
    sub-YYYY/
      ...
```
This does not follow a standard BIDS format. Having different annotation per site and modality this code can handle.

#### Structural Images

To create the 3D structural images on the Destrieux and Desikan-Killiany-Tourville (DKT) atlases run the follow Python command 

```Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
