"""
This module defines the basic training loop
for checking the SSL optimization having the
3D, 4D and dataloaders defined for all the possible
combination per modality following the combination Equation

\\binom{n}{2} = \\frac{n!}{2!(n-2)!} = \\frac{n(n-1)}{2}, and having n as 7 we will have
\\binom{7}{2} = \\frac{7!}{2!5!} = 21

as 21 possible pair combinations for the SSL optimization across
the entire dataset. All the seven possible combinations here

This Self-supervised learning (SSL) provides a training loop for ENIGMA-PTSD multimodal encoders.

This script instantiates a family of modality-specific encoders and runs a lightweight
forward-pass loop over the ENIGMA multimodal dataloader to validate:

 1) All modality tensors are produced correctly by the dataloader
 2) Each encoder consumes its expected input shape without shape/stride mismatches
 3) The 4D resting-state stream model (LateFusion4DResNet) can process a list of
  per-timepoint 3D volumes derived from the 4D fMRI input.

 Modalities handled (as implemented in this file):
  - RSdata: 4D resting-state fMRI -> split into T timepoint 3D streams -> LateFusion4DResNet
  - Structural maps: vol / surf / thick -> ResNet3DEncoder per modality
  - Functional maps: ALFF / fALFF / ReHo -> ResNet3DEncoder per modality

 Pairing context:
   In the full SSL setup, one can form all unique modality pairs for contrastive
   optimization, following C(n,2) = n(n-1)/2. With n=7 modalities, that yields 21
   possible pairs. This script focuses on validating forward passes for all encoders
   and preparing the embeddings needed by such pairwise losses.

 CLI arguments (via sys.argv):
     argv[1] time_samples : int
       Number of timepoints (streams) to sample/crop from the 4D RS fMRI for late fusion.
     argv[2] iterations : int
       Number of outer iterations to loop over the dataloader.
     argv[3] batch_size : int
       Batch size used by the ENIGMA dataloader.
     argv[4]
       learning rate as a float to train the SSL part

 call it like this: python train.py 66 100 10 1e-4

 Outputs:
    - Logs (via loguru) indicating successful reads and embedding computation.
    - Interim checkpoints or metrics are produced in the current script; this is a
      structural validation / debugging entrypoint.

 Notes:
 - Xavier initialization is applied across all encoders through init_xavier().
 - This script sets models to train() mode because SSL is typically trained with
   batch-dependent layers (e.g., dropout/norm); adjust to eval() for pure inference

"""

import sys

import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR

# get the dataloder and dataset definition here***
from ENIGMA_dataset_dataloder_creation_small import define_dataset_dataloader_ENIGMA

# get the model definitions here***
from ResNet_Encoders_definition import LateFusion4DResNet, ResNet3DEncoder

# import losses for SSL and alternative regularization
from losses_SSL import multimodal_pairwise_clip_loss

# define the device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialization function here
def init_xavier(m: nn.Module, uniform: bool = True):
    """
    Initialize learnable weights of a module using Xavier/Glorot initialization.

    This helper is designed to be used with 'nn.Module.apply(...)' so it will be
    called recursively on every submodule of a model. It targets the common layers
    used in this project:

    - nn.Linear
    - nn.Conv3d
    - nn.MultiheadAttention (handles packed in_proj_* and out_proj)
    - Normalization layers (LayerNorm, BatchNorm3d) are set to identity init

    Parameters
    ----------
    m : torch.nn.Module
      The submodule currently visited by 'Module.apply'.
    uniform : bool, default=True
      If True, uses 'nn.init.xavier_uniform_'.
      If False, uses 'nn.init.xavier_normal_'.

    --------
    - For Linear/Conv3d:
      - Weight initialized with Xavier (uniform or normal).
      - Bias (if present) is zeroed.
    - For MultiheadAttention:
      - Initializes 'in_proj_weight' (or equivalent packed QKV weights) with Xavier.
      - Zeroes 'in_proj_bias' if present.
      - Initializes 'out_proj.weight' with Xavier and zeroes 'out_proj.bias'.
    - For LayerNorm/BatchNorm3d:
      - weight <- ones, bias <- zeros (safe identity init)
    """

    # Linear
    if isinstance(m, nn.Linear):
        if uniform:
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Conv3d
    elif isinstance(m, nn.Conv3d):
        if uniform:
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # MultiheadAttention (handle its internal linear weights)
    elif isinstance(m, nn.MultiheadAttention):
        # in_proj (qkv packed) OR separate q/k/v weights
        if hasattr(m, "in_proj_weight") and m.in_proj_weight is not None:
            if uniform:
                nn.init.xavier_uniform_(m.in_proj_weight)
            else:
                nn.init.xavier_normal_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)

        # out projection
        if uniform:
            nn.init.xavier_uniform_(m.out_proj.weight)
        else:
            nn.init.xavier_normal_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)

    # Norm layers (usually keep default, but safe to set)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


def auto_padding(kernel, dilation=(1, 1, 1)):
    """
    Compute "same-ish" symmetric padding for 3D kernels under dilation.

    For odd-valued kernel sizes, this returns the padding that preserves spatial
    dimensions for stride=1 (approximately "same" convolution). The formula used:

      padding_i = dilation_i * (kernel_i // 2)

    Parameters
    ----------
    kernel : tuple[int, int, int]
      3D kernel size (kD, kH, kW). Expected to be odd in most cases.
    dilation : tuple[int, int, int], default=(1,1,1)
      Dilation factors along each dimension.

    Returns
    -------
    tuple[int, int, int]
      Padding values (pD, pH, pW) suitable for nn.Conv3d(..., padding=...).
    """
    return tuple(d * (k // 2) for k, d in zip(kernel, dilation, strict=False))


if __name__ == "__main__":
    """
      **Main section of the code**
    """

    # DEFINE HERE THE MODELS FOR EACH
    subject_indices_current_data = "../../Data/npz/subjects_overlaped_all_modalities.npz"

    time_samples = int(sys.argv[1])
    iterations = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])

    # define here the encoders for the fALFF/ReHo modalities
    enc_alff = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=128,
        emb_dim=256,
        attn_heads=1,
        activation_function="gelu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT event. Just
            # work on the convolutional kernels
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (3, 3, 3),
                "padding": auto_padding((3, 3, 3)),
            },
        ],
    )

    enc_falff = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=128,
        emb_dim=256,
        attn_heads=1,
        activation_function="gelu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT event. Just
            # work on the convolutional kernels
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (3, 3, 3),
                "padding": auto_padding((3, 3, 3)),
            },
        ],
    )

    enc_reho = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=128,
        emb_dim=256,
        attn_heads=1,
        activation_function="gelu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT event. Just
            # work on the convolutional kernels
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (3, 3, 3),
                "padding": auto_padding((3, 3, 3)),
            },
        ],
    )

    # define here the encoders for the structural modalities
    enc_vol = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=128,
        emb_dim=256,
        attn_heads=1,
        activation_function="silu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT event. Just
            # work on the convolutional kernels
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (1, 1, 1),
                "padding": auto_padding((1, 1, 1)),
            },
        ],
    )

    enc_surf = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=128,
        emb_dim=256,
        attn_heads=1,
        activation_function="silu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT event. Just
            # work on the convolutional kernels
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (1, 1, 1),
                "padding": auto_padding((1, 1, 1)),
            },
        ],
    )

    enc_thick = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=128,
        emb_dim=256,
        attn_heads=1,
        activation_function="silu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT event. Just
            # work on the convolutional kernels
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (1, 1, 1),
                "padding": auto_padding((1, 1, 1)),
            },
        ],
    )

    # define the 4D model for the RSData
    enc_4D_rsdata = LateFusion4DResNet(
        n_streams=time_samples,
        emb_dim=256,
        fusion="concat",
        out_dim=128,
        pretrained=False,
    )

    # initialize weights models
    for model in [enc_4D_rsdata, enc_alff, enc_falff, enc_reho, enc_vol, enc_surf, enc_thick]:
        model.to(device)
        model.apply(lambda m: init_xavier(m, uniform=True))


    # set the models in train mode
    enc_4D_rsdata.train()
    enc_alff.train()
    enc_falff.train()
    enc_reho.train()
    enc_vol.train()
    enc_surf.train()
    enc_thick.train()

    # define the optimizer here
    optimizer_SSL = torch.optim.AdamW(list(enc_4D_rsdata.parameters()) + list(enc_alff.parameters()) + list(enc_falff.parameters()) + list(enc_reho.parameters()) + list(enc_vol.parameters()) + list(enc_surf.parameters()) + list(enc_thick.parameters()), lr=learning_rate)
    scheduler_SSL = CosineAnnealingLR(optimizer_SSL, T_max=iterations, eta_min=1e-5)

    logger.info("All encoders has been defined!!")

    # GET HERE THE DATALODERS WITH THE PROJECTED IMAGES AND DIFFERENT MODALITIES - TAKING INTO ACCOUNT 21 DIFFERENT PAIRS
    # read the dataloder object here. Take 200s for all the trials/subjects here
    data_loader_ENIGMA = define_dataset_dataloader_ENIGMA(
        subject_indices_current_data=subject_indices_current_data,
        batch_size=batch_size,
        rs_time_window=200,
        rs_window_crop=time_samples,
        verbose=False
    )

    # define the modalities here
    # mods = ["RSdata", "vol", "surf", "thick", "alff", "falff", "reho"]
    # name_pairs = list(combinations(mods, 2))
    logger.info("Start SSL training!!..")

    for iter in range(0, iterations):
        # check if the dataloader works
        idx_sample = []
        sites_sample = []
        tr_values = []
        time_sub_values = []

        loss_interim = []

        for batch_data in data_loader_ENIGMA:
            if batch_data is None:  # validate this when batch is None and skip
                continue

            (
                idx,
                rs_DATA,
                st_DATA,
                falff_reho_DATA,
                subject_index,
                sites_idx,
                sampling_index,
                time_subject,
                TRs,
            ) = batch_data

            # define the data in cuda memory
            rs_DATA = rs_DATA.to(device, non_blocking=True)

            # if these are lists/tuples of tensors:
            st_DATA = [d.to(device, non_blocking=True) for d in st_DATA]
            falff_reho_DATA = [d.to(device, non_blocking=True) for d in falff_reho_DATA]

            # get the inputs on each modality and get the embeddings
            out_alff = enc_alff(falff_reho_DATA[0].unsqueeze(1))
            out_falff = enc_falff(falff_reho_DATA[1].unsqueeze(1))
            out_reho = enc_reho(falff_reho_DATA[2].unsqueeze(1))
            out_surf = enc_surf(st_DATA[0].unsqueeze(1))
            out_thick = enc_thick(st_DATA[1].unsqueeze(1))
            out_vol = enc_vol(st_DATA[2].unsqueeze(1))
            # create the list of tensor the 4D image timesamples
            rs_DATA = rs_DATA.permute(0, 4, 1, 2, 3).unsqueeze(2)
            RSDATA = [rs_DATA[:, t] for t in range(rs_DATA.shape[1])]
            out_rsdata = enc_4D_rsdata(RSDATA)

            embeds_all = {
              "alff":  out_alff,    # (B,D)
              "falff": out_falff,   # (B,D)
              "reho":  out_reho,    # (B,D)
              "surf":  out_surf,    # (B,D)
              "thick": out_thick,   # (B,D)
              "vol":   out_vol,     # (B,D)
              "rs":    out_rsdata,  # (B,D)
            }

            # define and update the loss here
            loss_SSL = multimodal_pairwise_clip_loss(embeds=embeds_all, temperature=0.07)
            loss_interim.append(loss_SSL)

            optimizer_SSL.zero_grad()
            loss_SSL.backward()
            optimizer_SSL.step()
            scheduler_SSL.step()

            # do the for across all the modalities
            tr_values.append(TRs)
            idx_sample.append(idx)
            time_sub_values.append(time_subject)
            sites_sample.append(sites_idx)
            # logger.info(f"Reading modalities for subject {subject_index}")
            # logger.success(f"batch size index {idx}")

        # report the interim loss here
        mean_loss = torch.stack([t.detach() for m in loss_interim]).mean()
        mean_loss_value = mean_loss.item()
        logger.info(f"Training iteration {iter} with loss: {mean_loss_value}..")
