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
     argv[5]
       type of SSL loss to apply across modalities
     argv[6]
       output_dim of each encoder in this experiment
     argv[7]
       selector of RS_encoder 0: ResNet with Fusion, 1: ResNet with LSTM
     argv[8]
       constant temperature of SSL optimization - InfoNCE loss (contrastive)

 call it like this: python train.py 66 100 10 1e-4 1 128 1 0.07

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
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import UMAP and t-SNE
import umap.umap_ as umap
from sklearn.manifold import TSNE

# import this for calculating metrics
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

# import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# get the dataloder and dataset definition here***
from dataset_dataloder.ENIGMA_dataset_dataloder_creation_small import define_dataset_dataloader_ENIGMA

# get the model definitions here***
from ResNet_Encoders_definition import LateFusion4DResNet, LateFusion4DResNet_LSTM, ResNet3DEncoder, LateFusion4DResNet_TemporalConv

# import losses for SSL and alternative regularization
from losses_SSL import multimodal_pairwise_clip_loss, multimodal_multipositive_infonce_whole

# define the device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set the seeds initialization here
seed = 42

# always define this seed before running your implementation!!
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# function for loading the latest checkpoint generated
def load_latest_ckpt(
    ckpt_dir: str,
    device: torch.device,
    enc_4D_rsdata,
    enc_alff,
    enc_falff,
    enc_reho,
    optimizer_SSL=None,
    scheduler_SSL=None,
):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    latest = ckpt_files[-1]
    logger.info(f"Loading latest checkpoint as {latest}")

    ckpt = torch.load(latest, map_location=device)

    # ---- models loading process here
    enc_4D_rsdata.load_state_dict(ckpt["models"]["enc_4D_rsdata"], strict=True)
    enc_alff.load_state_dict(ckpt["models"]["enc_alff"], strict=True)
    enc_falff.load_state_dict(ckpt["models"]["enc_falff"], strict=True)
    enc_reho.load_state_dict(ckpt["models"]["enc_reho"], strict=True)

    # ---- optimizer / scheduler loading - necessary for replication
    if optimizer_SSL is not None and "optimizer_SSL" in ckpt:
        optimizer_SSL.load_state_dict(ckpt["optimizer_SSL"])

    if scheduler_SSL is not None and "scheduler_SSL" in ckpt:
        scheduler_SSL.load_state_dict(ckpt["scheduler_SSL"])

    start_iter = int(ckpt.get("iter", 0)) + 1

    return start_iter, str(latest)

# function for reading the interim text files
def read_metric_txt(path_str: str):
    """
      reading the txt \n separated values
      for continuing with the training process
    """
    vals: list[float] = []
    for line in Path(path_str).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals

# function for plotting the interim metrics
def plotting_twinx_variables(time_vector, data1, data2, title: str, x_label: str, y_label1: str, y_label2: str, folder_images: str, iter: int):
    """
    plot her twinx the variables you want to compare
    across number of epochs in this case.

    Plot two time series on twin y-axes and save.

    Parameters
    ----------
    time_vector : array-like
    data1, data2 : array-like
        Series aligned to `time_vector`.
    title, x_label, y_label1, y_label2 : str
    folder_images : str
    subj : str

    Returns
    -------
    None
    """

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_vector, data1, "b-", label=y_label1, linewidth=3)
    ax1.set_xlabel(x_label, fontsize=1)
    ax1.set_ylabel(y_label1, color="blue", fontsize=16)
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()

    ax2.plot(time_vector, data2, "r-", label=y_label2, linewidth=3)
    ax2.set_ylabel(y_label2, color="red", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.grid(True)

    for tick in ax1.get_xticklabels():
        tick.set_fontsize(14)

    plt.title(title)
    fig.legend()
    fig.savefig(f"{folder_images}/{y_label1}_{y_label2}_{iter}.jpg")
    plt.close("all")

# function for define the metrics here..
def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def wasserstein_avg_over_dims(A: np.ndarray, B: np.ndarray) -> float:
    """
    A, B: (B, D) arrays.
    Computes 1D Wasserstein distance per dimension across the batch, then averages.
    """
    assert A.shape == B.shape, (A.shape, B.shape)
    D = A.shape[1]
    return float(np.mean([wasserstein_distance(A[:, d], B[:, d]) for d in range(D)]))


def pairwise_wasserstein(embeds_np: dict) -> dict:
    """
    embeds_np: dict name -> (B,D)
    returns dict (name_i, name_j) -> wasserstein distance
    """
    keys = list(embeds_np.keys())
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            out[(ki, kj)] = wasserstein_avg_over_dims(embeds_np[ki], embeds_np[kj])
    return out


def stack_modalities(embeds_np: dict):
    """
    Returns:
      X: (M*B, D)
      modality_labels: (M*B,) integers 0..M-1
    """
    keys = list(embeds_np.keys())
    X = np.concatenate([embeds_np[k] for k in keys], axis=0)
    B = embeds_np[keys[0]].shape[0]
    modality_labels = np.concatenate([np.full(B, i, dtype=int) for i in range(len(keys))], axis=0)
    return X, modality_labels, keys


def compute_nmi_via_kmeans(X: np.ndarray, true_labels: np.ndarray, n_clusters: int, seed: int = 42) -> float:
    """
    Cluster X, then compute NMI between cluster assignments and provided labels.
    """
    true_labels = np.asarray(true_labels)
    uniq = np.unique(true_labels)
    if uniq.size < 2:
        return float("nan")  # or 0.0 if you prefer
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    pred = km.fit_predict(X)
    return float(normalized_mutual_info_score(true_labels, pred))

def compute_silhouette(X: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> float:
    """
    Silhouette score for given labels.
    With normalized embeddings, cosine is usually a good choice.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan")  # or 0.0 if you prefer
    return float(silhouette_score(X, labels, metric=metric))

def make_subject_palette(subject_ids, palette_name="tab20"):
    """
    Create a deterministic subject->color mapping.
    subject_ids: iterable of subject labels (strings or ints)
    """
    uniq = list(dict.fromkeys(subject_ids))  # preserves first-seen order
    colors = sns.color_palette(palette_name, n_colors=len(uniq))
    return {sid: col for sid, col in zip(uniq, colors)}

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

def get_concat_embedding(rs_DATA, falff_reho_DATA,
                        enc_alff, enc_falff, enc_reho,
                        enc_4D_rsdata, batch_sizes):
    """
      get all the embeddings from each iteration here
      project all the data with the models trained after that
    """
    out_alff  = enc_alff(falff_reho_DATA[0].unsqueeze(1))
    out_falff = enc_falff(falff_reho_DATA[1].unsqueeze(1))
    out_reho  = enc_reho(falff_reho_DATA[2].unsqueeze(1))
    # out_surf  = enc_surf(st_DATA[0].unsqueeze(1))
    # out_thick = enc_thick(st_DATA[1].unsqueeze(1))
    # out_vol   = enc_vol(st_DATA[2].unsqueeze(1))

    rs_ = rs_DATA.permute(0, 4, 1, 2, 3).unsqueeze(2)  # (B,T,1,D,H,W)
    RSDATA = [rs_[:, t] for t in range(rs_.shape[1])]
    out_rs = enc_4D_rsdata(RSDATA)

    embeds_all = {
        "alff": out_alff, "falff": out_falff, "reho": out_reho, "rs": out_rs
    }
    embeds_all = {k: F.normalize(v, dim=1) for k, v in embeds_all.items()}

    # stack this in to_np way
    embeds_np = {k: to_np(v) for k, v in embeds_all.items()}

    wd = pairwise_wasserstein(embeds_np)

    X_all, mod_labels, mod_keys = stack_modalities(embeds_np)

    val_clust = np.tile(np.arange(batch_sizes), 4)

    if len(val_clust) > 1:
       sil_mod = compute_silhouette(X_all, val_clust, metric="cosine")
       nmi_mod = compute_nmi_via_kmeans(X_all, val_clust, n_clusters=batch_sizes)
    else:
       sil_mod = np.nan
       nmi_mod = np.nan

    z = torch.cat([embeds_all[k] for k in ["alff","falff","reho","rs"]], dim=1)
    return z, wd, sil_mod, nmi_mod, embeds_all

## get the UMAP and tSNE projections here..
def get_UMAP_tSNE_projections(Z, proj_components:int=2, ndim:int=128):
    """
      get the UMAP and TSNE projects
      given the eval embeddings
    """
    umap_map = umap.UMAP(n_neighbors=5, min_dist=0.1, spread=3.0, n_components=proj_components, random_state=42, verbose=True, metric="euclidean")
    # take into account these values are concat projections of ndim values
    # reshape here the input values here
    Z_stacked = np.vstack(Z)
    Z_reshaped = Z_stacked.reshape(Z_stacked.shape[0], 4, ndim)
    Z_final = Z_reshaped.reshape(-1, ndim)
    umap_proj = umap_map.fit_transform(Z_final)

    return umap_proj

# plot the projections here
def plot_proj_feat(proj_feat, sub_labels, subject_order, subject_palette, modalities, title: str, epoch: str, suffix: str, folder_name: str):
    """
    Scatter-plot 2D t-SNE embeddings and save to disk.

    Parameters
    ----------
    proj_feat : np.ndarray, shape (N, 2) -> this can be tsne_feat or umap_feat input representations
    labels : array-like
        Class/subject labels for coloring.
    title, epoch, suffix, folder_name, class_value, subj : str
        Strings for titling, legend label, and output filename.

    Returns
    -------
    None
    """

    markers=["o", "X", "s", "D"]
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=proj_feat[:, 0], y=proj_feat[:, 1], hue=sub_labels, hue_order=subject_order, style=modalities, markers=markers, palette=subject_palette, s=180, edgecolor="k", alpha=0.7, legend=False)
    plt.title(title)
    # plt.legend(title=class_value, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{folder_name}/{suffix}_iter_{epoch}.jpg")
    plt.close("all")

if __name__ == "__main__":
    """
      **Main section of the code**
    """

    # DEFINE HERE THE MODELS FOR EACH
    subject_indices_current_data = "/home/mayortorresjm/ENIGMA-PTSD/Data/npz/subjects_overlaped_all_modalities.npz"

    time_samples = int(sys.argv[1])
    iterations = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    loss_selector = int(sys.argv[5])
    out_dim = int(sys.argv[6])
    rs_data_model_sel = int(sys.argv[7])
    temperature = float(sys.argv[8])

    model_path = f"./models/folder_4_{iterations}_{batch_size}_{learning_rate}_{loss_selector}_{out_dim}_{rs_data_model_sel}_{temperature}"
    vis_path = f"./visualization/folder_4_{iterations}_{batch_size}_{learning_rate}_{loss_selector}_{out_dim}_{rs_data_model_sel}_{temperature}"
    # Define here the models and plot folders for interim results visualization
    if not  os.path.exists(model_path):
       os.makedirs(model_path, exist_ok=True)
    if not os.path.exists(vis_path):
       os.makedirs(vis_path, exist_ok=True)

    # define here the encoders for the fALFF/ReHo modalities
    enc_alff = ResNet3DEncoder(
        in_channels=1,
        # interrupting the action of some layers if conside or not
        use_stages=(True, False, False, False),
        out_dim=out_dim,
        emb_dim=out_dim*2,
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
        out_dim=out_dim,
        emb_dim=out_dim*2,
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
        out_dim=out_dim,
        emb_dim=out_dim*2,
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
        out_dim=out_dim,
        emb_dim=out_dim*2,
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
        out_dim=out_dim,
        emb_dim=out_dim*2,
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
        out_dim=out_dim,
        emb_dim=out_dim*2,
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
    if rs_data_model_sel == 0:
       enc_4D_rsdata = LateFusion4DResNet(
           n_streams=time_samples,
           emb_dim=out_dim*2,
           fusion="concat",
           out_dim=out_dim,
           pretrained=False,
       )
    elif rs_data_model_sel == 1:
       enc_4D_rsdata = LateFusion4DResNet_LSTM(
           n_streams=time_samples,
           emb_dim=out_dim*2,
           fusion="lstm",          # kept for compatibility; you can ignore
           out_dim=int(out_dim/2),             # each ResNet3DEncoder output dim
           pretrained=False,
           lstm_hidden=int(out_dim/2),
           lstm_layers=2,
           bidirectional=True,
           lstm_dropout= 0.75,
           pool= "mean",            # "last" | "mean"
           out_dim_final=out_dim,
       )
    else:
       enc_4D_rsdata = LateFusion4DResNet_TemporalConv(
           n_streams=time_samples,
           emb_dim=out_dim*2,
           out_dim=out_dim,            # per-branch output channels C
           pretrained=False,
           kernel_size=5,
           n_blocks=4,
           hidden_channels=100,
           dilation_growth=2,
           attention_projection=True,
           attn_heads=1,
           pool="mean",
           out_dim_final=out_dim,
       )

    # initialize weights models
    for model in [enc_4D_rsdata, enc_alff, enc_falff, enc_reho]:
        model.to(device)
        model.apply(lambda m: init_xavier(m, uniform=True))

    # define the optimizer here
    optimizer_SSL = torch.optim.AdamW(list(enc_4D_rsdata.parameters()) + list(enc_alff.parameters()) + list(enc_falff.parameters()) + list(enc_reho.parameters()), lr=learning_rate)
    scheduler_SSL = CosineAnnealingLR(optimizer_SSL, T_max=iterations, eta_min=1e-5)

    logger.info("All encoders has been defined!!")

    if os.path.exists(model_path):
       start_iter, _ = load_latest_ckpt(model_path, device, enc_4D_rsdata, enc_alff, enc_falff, enc_reho, optimizer_SSL, scheduler_SSL)
       MI_vals = read_metric_txt(f"{model_path}/mutual_information_interim.txt")
       SIL_vals = read_metric_txt(f"{model_path}/silhoutte_interim.txt")
       WD_vals = read_metric_txt(f"{model_path}/wasserstein_distance_interim.txt")
       LOSS_vals = read_metric_txt(f"{model_path}/losses_interim.txt")
       ITERS = list(range(0, start_iter, 10))
    else:
       start_iter = 0
       MI_vals = []
       SIL_vals = []
       WD_vals = []
       ITERS = []
       LOSS_vals = []

    # set the models in train mode
    enc_4D_rsdata.train()
    enc_alff.train()
    enc_falff.train()
    enc_reho.train()

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

    if loss_selector == 0:
       logger.info("Using SSL loss per modality pairs")
    else:
       logger.info("Using SSL loss across ALL the modalities")

    for iter in range(start_iter, iterations):

        # check if the dataloader works
        idx_sample = []
        sites_sample = []
        tr_values = []
        time_sub_values = []
        subjects_ids = []

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

            # if these are lists/tuples of tensors for st_DATA and falff_reho_DATA:
            st_DATA = [d.to(device, non_blocking=True) for d in st_DATA]
            falff_reho_DATA = [d.to(device, non_blocking=True) for d in falff_reho_DATA]

            # get the inputs on each modality and get the embeddings
            out_alff = enc_alff(falff_reho_DATA[0].unsqueeze(1))
            out_falff = enc_falff(falff_reho_DATA[1].unsqueeze(1))
            out_reho = enc_reho(falff_reho_DATA[2].unsqueeze(1))
            #out_surf = enc_surf(st_DATA[0].unsqueeze(1))
            #out_thick = enc_thick(st_DATA[1].unsqueeze(1))
            #out_vol = enc_vol(st_DATA[2].unsqueeze(1))
            # create the list of tensor the 4D image timesamples. Do this permutation before transforming the tensor to a list of tensors!!
            rs_DATA = rs_DATA.permute(0, 4, 1, 2, 3).unsqueeze(2)
            RSDATA = [rs_DATA[:, t] for t in range(rs_DATA.shape[1])]
            out_rsdata = enc_4D_rsdata(RSDATA)

            embeds_all = {
              "alff":  out_alff,    # (B,D)
              "falff": out_falff,   # (B,D)
              "reho":  out_reho,    # (B,D)
              "rs":    out_rsdata,  # (B,D)
            }

            # normalize the embeddings here to make it easier map in the l2-sphere later..
            embeds_all = {k_embeds: F.normalize(v_embeds, dim=1) for k_embeds, v_embeds in embeds_all.items()}

            # define and update the loss here. Use the current embeddings to calculate the loss but NOT the ones for the clustering representation
            if loss_selector == 0:
               loss_SSL = multimodal_pairwise_clip_loss(embeds=embeds_all, temperature=temperature)
            else:
               loss_SSL = multimodal_multipositive_infonce_whole(embeds=embeds_all, temperature=temperature)

            loss_interim.append(loss_SSL)

            # do the loss backwards and the zero grad update
            optimizer_SSL.zero_grad()
            loss_SSL.backward()
            optimizer_SSL.step()
            scheduler_SSL.step()

            # do the for across all the modalities
            tr_values.append(TRs)
            idx_sample.append(idx)
            time_sub_values.append(time_subject)
            sites_sample.append(sites_idx)
            subjects_ids.append(subject_index)
            # logger.info(f"Reading modalities for subject {subject_index}")
            # logger.success(f"batch size index {idx}")

        # get the embeddings projected in eval mode here
        # report the interim loss here
        mean_loss = torch.stack([m.detach() for m in loss_interim]).mean()
        mean_loss_value = mean_loss.item()
        logger.info(f"Training iteration {iter} with loss: {mean_loss_value}..")

        models = [enc_4D_rsdata, enc_alff, enc_falff, enc_reho] # enc_vol, enc_surf, enc_thick]
        # processing the projected embeddings in eval mode
        # ---- END OF EPOCH: EMBEDDING SNAPSHOT IN EVAL MODE ----
        # (optional: only every few epochs)
        Z_vals = []
        sites_evals = []
        subs_evals = []
        mi_vals = []
        sil_vals = []
        wd_vals = []

        if iter == 0 or iter % 10 == 0:
           prev_modes = [m.training for m in models]
           for model in models:
               model.eval()

           # collect a limited number of batches to keep it light
           with torch.no_grad():
               for j, batch_data_eval in enumerate(data_loader_ENIGMA):

                   # read all the batches again to project the embeddings in eval mode**
                   (idx_eval, rs_DATA_eval, st_DATA_eval, falff_reho_DATA_eval, subject_index_eval, sites_idx_eval,
                   sampling_index_eval, time_subject_eval, TRs_eval) = batch_data_eval

                   rs_DATA_eval = rs_DATA_eval.to(device, non_blocking=True)
                   # st_DATA_eval = [d.to(device, non_blocking=True) for d in st_DATA_eval]
                   falff_reho_DATA_eval = [d.to(device, non_blocking=True) for d in falff_reho_DATA_eval]

                   z, wd, sil, nmi, _ = get_concat_embedding(
                      rs_DATA_eval, falff_reho_DATA_eval,
                      enc_alff, enc_falff, enc_reho,
                      enc_4D_rsdata, len(idx_eval)
                   )

                   Z_vals.append(z.detach().cpu().numpy())
                   sites_evals.append(sites_idx_eval)
                   subs_evals.append(subject_index_eval)

                   mi_vals.append(nmi)
                   sil_vals.append(sil)
                   wd_vals.append(sum(wd.values()) / len(wd))

           # get UMAP and t-SNE projections
           umap_projections = get_UMAP_tSNE_projections(Z=Z_vals, proj_components=2, ndim=out_dim)
           modality_ids = np.tile(np.arange(4), np.vstack(Z_vals).shape[0])
           subs_evals = [str(s) for sublist in subs_evals for s in sublist]
           subs_evals = np.array(subs_evals)  # convert list → array
           subs_evals_long = np.repeat(subs_evals, 4)

           MI_vals.append(np.nanmean(np.array(mi_vals)))
           SIL_vals.append(np.nanmean(np.array(sil_vals)))
           WD_vals.append(np.nanmean(np.array(wd_vals)))
           LOSS_vals.append(mean_loss_value)
           ITERS.append(iter)

           ## append the values in the last part of the metrics file
           with open(f"{model_path}/mutual_information_interim.txt", "a") as f_mi:
              f_mi.write(str(np.nanmean(np.array(mi_vals))) + '\n')
           with open(f"{model_path}/wasserstein_distance_interim.txt", "a") as f_wd:
              f_wd.write(str(np.nanmean(np.array(wd_vals))) + '\n')
           with open(f"{model_path}/silhoutte_interim.txt", "a") as f_sil:
              f_sil.write(str(np.nanmean(np.array(sil_vals))) + '\n')
           with open(f"{model_path}/losses_interim.txt", "a") as f_loss:
              f_loss.write(str(mean_loss_value) + '\n')

           # plotting the metrics here
           logger.info("Plotting metrics for iteration {iter}!!")
           plotting_twinx_variables(np.array(ITERS), np.array(LOSS_vals), np.array(MI_vals), title=f"SSL loss and MI", x_label="epochs", y_label1="SSL Loss", y_label2="NMI", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(LOSS_vals), np.array(SIL_vals), title=f"SSL loss and Silhoutte", x_label="epochs", y_label1="SSL Loss", y_label2="Silhoutte", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(LOSS_vals), np.array(WD_vals), title=f"SSL loss and Wasserstein Distance", x_label="epochs", y_label1="SSL Loss", y_label2="Wasserstein Distance", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(MI_vals), np.array(WD_vals), title=f"MI and Wasserstein Distance", x_label="epochs", y_label1="NMI", y_label2="Wasserstein Distance", folder_images=vis_path, iter=iter)
           plotting_twinx_variables(np.array(ITERS), np.array(MI_vals), np.array(SIL_vals), title=f"MI and Silhoutte", x_label="epochs", y_label1="NMI", y_label2="Silhoutte", folder_images=vis_path, iter=iter)

           if iter == 0 or iter == start_iter + 9:
              subject_order = sorted(np.unique(subs_evals_long).tolist())
              subject_palette = make_subject_palette(subject_order, palette_name="tab20")

           logger.info(f"Plotting UMAP projections for iteration {iter}!!")
           # plotting the UMAP projections here
           plot_proj_feat(proj_feat=umap_projections, sub_labels=subs_evals_long, subject_order=subject_order, subject_palette=subject_palette, modalities=modality_ids, title=f"UMAP projections for epoch {iter} - SSL loss {mean_loss_value}", epoch=str(iter), suffix="umap_projections", folder_name=vis_path)

           # restore to train modes back
           for model, was_train in zip(models, prev_modes):
               model.train(was_train)

           # save the checkpoint here
           ckpt_path = os.path.join(model_path, f"ssl_ckpt_iter_{iter:05d}.pth")

           ckpt = {
                "iter": iter,
                "models": {
                   "enc_4D_rsdata": enc_4D_rsdata.state_dict(),
                   "enc_alff": enc_alff.state_dict(),
                   "enc_falff": enc_falff.state_dict(),
                   "enc_reho": enc_reho.state_dict(),
                },
                "optimizer_SSL": optimizer_SSL.state_dict(),
                "scheduler_SSL": scheduler_SSL.state_dict(),
           }

           torch.save(ckpt, ckpt_path)
           logger.success(f"Saving checkpoint for iteration {iter}!!")

