"""
   Module for testing
   the labels pairing for
   each pretrained encoder
   across, sites, TR or Age labels
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
import sys
import numpy as np
import warnings

# To ignore all warnings:
warnings.filterwarnings("ignore")
# To ignore specific categories of warnings (e.g., DeprecationWarning):
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import umap.umap_ as umap

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import (load_latest_ckpt_4_mod,
                   read_metric_txt,
                   plotting_twinx_variables,
                   to_np,
                   make_subject_palette,
                   make_site_palette,
                   wasserstein_avg_over_dims,
                   pairwise_wasserstein,
                   stack_modalities,
                   init_xavier,
                   compute_silhouette,
                   compute_nmi_via_kmeans,
                   auto_padding
                   )

from ResNet_Encoders_definition import LateFusion4DResNet, LateFusion4DResNet_LSTM, ResNet3DEncoder, LateFusion4DResNet_TemporalConv
from dataset_dataloder.ENIGMA_dataset_dataloder_creation_small import define_dataset_dataloader_ENIGMA
from sklearn.manifold import TSNE

# define the device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# import the utils function from tehe utils module
# get the dataloder and dataset definition here***
# get the model definitions here***

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


"""
  Define the main section of the code here
"""

if __name__ == "__main__":
    """
        **Main section of the code**
    """

    # receive the model path here as string for interim reading**
    time_samples = int(sys.argv[1])
    iterations = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    loss_selector = int(sys.argv[5])
    out_dim = int(sys.argv[6])
    rs_data_model_sel = int(sys.argv[7])
    temperature = float(sys.argv[8])
    iteration_ckpth = int(sys.argv[9])

    model_path = f"./models/folder_4_{iterations}_{batch_size}_{learning_rate}_{loss_selector}_{out_dim}_{rs_data_model_sel}_{temperature}"
    out_path = f"./visualization_test_projections/folder_4_{iterations}_{batch_size}_{learning_rate}_{loss_selector}_{out_dim}_{rs_data_model_sel}_{temperature}"

    subject_indices_current_data = "/home/mayortorresjm/ENIGMA-PTSD/Data/npz/subjects_overlaped_all_modalities.npz"


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

    # define here the encoders for the structural modalities
    enc_vol = ResNet3DEncoder(
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

    enc_surf = ResNet3DEncoder(
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


    enc_thick = ResNet3DEncoder(
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

    optimizer_SSL = torch.optim.AdamW(list(enc_4D_rsdata.parameters()) + list(enc_alff.parameters()) + list(enc_falff.parameters()) + list(enc_reho.parameters()), lr=learning_rate)
    scheduler_SSL = CosineAnnealingLR(optimizer_SSL, T_max=iterations, eta_min=1e-5)

    # load the desired checpoint
    iter, _ = load_latest_ckpt_4_mod(model_path, device, enc_4D_rsdata, enc_alff, enc_falff, enc_reho, optimizer_SSL, scheduler_SSL, iteration_ckpth)

    for model in [enc_reho, enc_falff, enc_alff, enc_4D_rsdata]:
        model.to(device)

    # define the dataloader here with shuffling in false for replication
    # REMEMBER THIS IS A NON-CONVEX PROBLEM AND IN TRAINING THE ORDER WILL
    # ALWAYS MATTER, ESPECIALLY TO GET FULL-REPLICABLE RESULTS!!
    data_loader_ENIGMA = define_dataset_dataloader_ENIGMA(
        subject_indices_current_data=subject_indices_current_data,
        batch_size=batch_size,
        rs_time_window=200,
        rs_window_crop=time_samples,
        verbose=False,
        shuffling=False
    )

    # create the output path if necessary**
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # first read the older models as a checkpoint here just projecting the
    # latent space
    MI_vals = read_metric_txt(f"{model_path}/mutual_information_interim.txt")
    SIL_vals = read_metric_txt(f"{model_path}/silhoutte_interim.txt")
    WD_vals = read_metric_txt(f"{model_path}/wasserstein_distance_interim.txt")
    LOSS_vals = read_metric_txt(f"{model_path}/losses_interim.txt")
    ITERS = list(range(0, iter, 10))

    models = [enc_vol, enc_thick, enc_surf]
    # processing the projected embeddings in eval mode
    # ---- END OF EPOCH: EMBEDDING SNAPSHOT IN EVAL MODE ----
    # (optional: only every few epochs)
    Z_vals = []
    sites_evals = []
    age_evals = []
    subs_evals = []
    mi_vals = []
    sil_vals = []
    wd_vals = []

    prev_modes = [m.training for m in models]
    for model in models:
        model.eval()

    age_thr = 25

    # collect a limited number of batches to keep it light
    with torch.no_grad():
        for j, batch_data_eval in enumerate(data_loader_ENIGMA):

            if batch_data_eval is None:  # validate this when batch is None and skip
                continue

            # read all the batches again to project the embeddings in eval
            # mode**

            (idx_eval,
             rs_DATA_eval,
             st_DATA_eval,
             falff_reho_DATA_eval,
             subject_index_eval,
             sites_idx_eval,
             sampling_index_eval,
             time_subject_eval,
             TRs_eval,
             age_val) = batch_data_eval

            rs_DATA_eval = rs_DATA_eval.to(device, non_blocking=True)
            #st_DATA_eval = [d.to(device, non_blocking=True)
            #                for d in st_DATA_eval]
            # falff_reho_DATA_eval = [d.to(device, non_blocking=True) for d in falff_reho_DATA_eval]
            falff_reho_DATA_eval = [d.to(device, non_blocking=True) for d in falff_reho_DATA_eval]

            z, wd, sil, nmi, _ = get_concat_embedding(
                      rs_DATA_eval, falff_reho_DATA_eval,
                      enc_alff, enc_falff, enc_reho,
                      enc_4D_rsdata, len(idx_eval)
            )

            Z_vals.append(z.detach().cpu().numpy())
            sites_evals.append(sites_idx_eval)
            subs_evals.append(subject_index_eval)
            age_evals.append(age_val)

            mi_vals.append(nmi)
            sil_vals.append(sil)
            wd_vals.append(sum(wd.values()) / len(wd))


    MI_vals = np.nanmean(np.array(mi_vals))
    SIL_vals = np.nanmean(np.array(sil_vals))
    WD_vals = np.nanmean(np.array(wd_vals))

    # do the list arrangement per SUBJECT here
    # define here the subject list for plotting subsequently...
    subs_evals = [str(s) for sublist in subs_evals for s in sublist]
    subs_evals = np.array(subs_evals)  # convert list → array
    # multiply by two given the two modalities in this experiment, vary it if
    # this necessary..
    subs_evals_long = np.repeat(subs_evals, 4)

    # get the palette here
    subject_order = sorted(np.unique(subs_evals_long).tolist())
    subject_palette = make_subject_palette(subject_order, palette_name="tab20")

    # do the list arrangement per site
    # define here the subject list for plotting subsequently...
    sites_evals = [str(s) for sublist in sites_evals for s in sublist]
    sites_evals = np.array(sites_evals)  # convert list → array
    # multiply by two given the two modalities in this experiment, vary it if
    # this necessary..
    sites_evals_long = np.repeat(sites_evals, 4)

    # get the palette here
    sites_order = sorted(np.unique(sites_evals_long).tolist())
    sites_palette = make_site_palette(sites_order, palette_name="tab20")

    # age_tensor = torch.stack(age_evals)          # shape: (N,)
    age_all = torch.cat([t.reshape(-1) for t in age_evals], dim=0) #(age_tensor >= age_thr).long().cuda() #age_binary = [1 if x >= age_thr else 0 for x in age_evals] #(age_evals >= age_thr).astype(int)
    age_binary = ((age_all >= age_thr).long().cuda()).cpu().numpy()

    # get the palette here
    age_order = ["young", "adult"] #sorted(np.unique(age_binary).tolist())
    age_palette = make_subject_palette(age_order, palette_name="tab10")
    age_binary_long = np.repeat(age_binary, 4)
    age_labels = np.where(age_binary_long == 0, "young", "adult")

    # get UMAP and t-SNE projections
    umap_projections = get_UMAP_tSNE_projections(
        Z=Z_vals, proj_components=2, ndim=out_dim)
    modality_ids = np.tile(np.arange(4), np.vstack(Z_vals).shape[0])

    # define the other flatten lits HERE for plotting subsequently**
    logger.info(f"Plotting UMAP projections in subject level!!")
    # plotting the UMAP projections here per subject
    plot_proj_feat(
        proj_feat=umap_projections,
        sub_labels=subs_evals_long,
        subject_order=subject_order,
        subject_palette=subject_palette,
        modalities=modality_ids,
        title=f"UMAP projections subject level",
        epoch=str(iter-1),
        suffix=f"umap_projections_subject",
        folder_name=out_path)
    # define the other flatten lits HERE for plotting subsequently**
    logger.info(f"Plotting UMAP projections in site level!!")
    # plotting the UMAP projections here per subject
    plot_proj_feat(
        proj_feat=umap_projections,
        sub_labels=sites_evals_long,
        subject_order=sites_order,
        subject_palette=sites_palette,
        modalities=modality_ids,
        title=f"UMAP projections site level",
        epoch=str(iter-1),
        suffix=f"umap_projections_site",
        folder_name=out_path, legend=True)


    # define the other flatten lits HERE for plotting subsequently**
    logger.info(f"Plotting UMAP projections age labels young-adult!!")
    # plotting the UMAP projections here per subject
    plot_proj_feat(
        proj_feat=umap_projections,
        sub_labels=age_labels,
        subject_order=age_order,
        subject_palette=age_palette,
        modalities=modality_ids,
        title=f"UMAP projections age labels",
        epoch=str(iter-1),
        suffix=f"umap_projections_age",
        folder_name=out_path, legend=True)
