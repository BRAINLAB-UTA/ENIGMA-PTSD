"""
   Module for adding utils function
   for SSL training and dataloader modules
   functionalities
"""
import sys
import os

import torch
import torch.nn as nn

from pathlib import Path
import numpy as np

# import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from loguru import logger

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
    enc_thick,
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
    enc_thick.load_state_dict(ckpt["models"]["enc_thick"], strict=True)

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
    ax1.set_xlabel(x_label, fontsize=16)
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
