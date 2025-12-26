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
   Codes for pre-processing, dataloader creating, and Self-Supervised Learning evaluation over current ENIGMA-PTSD dataset. For accessing ENIGMA-PTSD data please reach professor Xi Zhu, PhD in this [email](xi.zhu@uta.edu).
</p>

This repository contains:
- **Preprocessing pipelines** to generate in this [folder](https://github.com/BRAINLAB-UTA/ENIGMA-PTSD/tree/main/preprocessing):
  - structural features (sMRI-derived)
  - fALFF/ReHo (rs-fMRI-derived)
  - RSData (resting-state 4D time-series–derived)
- **Dataloader creation utilities** for downstream SSL evaluation/training in this [folder](https://github.com/BRAINLAB-UTA/ENIGMA-PTSD/tree/main/SSL_evaluation).
