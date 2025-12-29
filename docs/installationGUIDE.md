Installation Guide (Mixed OS)

This guide provides installation steps for the **ENIGMA-PTSD repo (Python 3.11)**, **FSL (with FSLeyes)**, and **FreeSurfer** on Linux, macOS, and Windows (via WSL2).

---

## 1) Linux (Ubuntu / Debian)

### A) Python 3.11 environment for the repository

```bash
conda create -n enigma-ptsd-py311 python=3.11 -y
conda activate enigma-ptsd-py311
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### B) Install FSL (Full stack, official installer)

Take the Python from the FSL official page:

```bash
wget -O fslinstaller.py https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py
python fslinstaller.py
```

Add FSL to PATH (restart terminal after) using the source file [here](https://github.com/BRAINLAB-UTA/ENIGMA-PTSD/blob/main/fsl_load.sh), be sure you are exporting the right absolute path with the FSL bin files in your machine.


Launch FSLeyes (already bundled with FSL):

```bash
fsleyes
```

### C) Install FreeSurfer (if needed)

Download and install FreeSurfer Linux version:

```bash
wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz
tar -xvzf freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz -C $HOME
```

Configure FreeSurfer:

```bash
export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
which recon-all
```

Add license - First obtain the license by email [here](https://surfer.nmr.mgh.harvard.edu/registration.html):

```bash
echo "PASTE_YOUR_FREESURFER_LICENSE_HERE" > $FREESURFER_HOME/license.txt
```

---

## 2) macOS (Intel or Apple Silicon)

### A) First install miniconda or condaforge doing this

```bash
curl -O repo.anaconda.com
```

and install the .sh insaller doing this and following the instruction (be sure about the location of the bin files)

```bash
bash Miniconda3-latest-MacOSX-x86_64.sh
```

Follow the same instructions from the Linux side except for the freesufer parth

### B) Install freesurfer on mac

Download and install FreeSurfer MacOSX version:

```bash
wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-macOS-darwin_x86_64-7.4.1.tar.gz
tar -xvzf freesurfer-macOS-darwin_x86_64-7.4.1.tar.gz -C $HOME
```

Configure FreeSurfer in MacOSX:

```bash
export FREESURFER_HOME=/Applications/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
which recon-all
```
 
## 3) Windows 11/10 (Recommended: WSL2)

### A) Install WSL2 + Ubuntu

Open PowerShell terminal as Command Prompt as administrator and run:


```bash
wsl --install
```

Now run the same instruction defined in the Linux section [here](#1-linux-ubuntu--debian)





































