"""
  Losses definition SSL
  and possibilities for regularization
"""

import torch
import torch.nn.functional as F
from itertools import combinations


# dfined here the wassersetein distance by pair for latter define a barycenter
def sliced_wasserstein_loss(z_a, z_b, n_projections=64, p=2, normalize=True):
    assert z_a.shape == z_b.shape
    N, D = z_a.shape

    if normalize:
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

    proj = torch.randn(n_projections, D, device=z_a.device, dtype=z_a.dtype)
    proj = F.normalize(proj, dim=1)

    a_proj = z_a @ proj.t()   # (N, P)
    b_proj = z_b @ proj.t()   # (N, P)

    a_sorted, _ = torch.sort(a_proj, dim=0)
    b_sorted, _ = torch.sort(b_proj, dim=0)

    diff = a_sorted - b_sorted
    if p == 1:
        return diff.abs().mean()
    elif p == 2:
        return (diff ** 2).mean()
    else:
        return (diff.abs() ** p).mean()

# add orthogonal penalty to the sites label here. This must be done to avoid the site as possible confound
def orthogonality_penalty(
     Z: torch.Tensor,
     ids: torch.Tensor,
     n_domain: int,
     normalize: bool = True,
    ):
    """
    Orthogonality penalty: || Z^T S ||_F^2

    Args
    ----
    Z : (N, D) tensor
        Embeddings
    site_ids : (N,) tensor (long)
        Site labels (0 ... n_sites-1)
    n_domain : int
        Number of domains per batch
    normalize : bool
        Whether to center/normalize Z and S (recommended)

    Returns
    -------
    loss : scalar tensor
        Site orthogonality penalty
    """

    N, D = Z.shape

    # One-hot encode domain labels → (N, K)
    S = F.one_hot(ids, num_classes=n_domain).float()

    if normalize:
        # Center embeddings
        Z = Z - Z.mean(dim=0, keepdim=True)
        Z = Z / (Z.std(dim=0, keepdim=True) + 1e-6)

        # Center site indicators
        S = S - S.mean(dim=0, keepdim=True)

    # Z^T S → (D, K)
    ZtS = Z.T @ S

    # Frobenius norm squared for extending the optimization
    loss = torch.sum(ZtS ** 2) / (D * n_domain)

    return loss

def l2norm(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def clip_loss_pair(z_a, z_b, temperature=0.07):
    # z_a, z_b: (B,D)
    z_a = l2norm(z_a)
    z_b = l2norm(z_b)

    logits_ab = (z_a @ z_b.T) / temperature
    logits_ba = (z_b @ z_a.T) / temperature

    # they are not labels related to subject-level diagnosis or condition - This labels are aranged by the positive (aligned)
    # pairs per batch!
    aranged_gen_labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss = 0.5 * (F.cross_entropy(logits_ab, aranged_gen_labels) + F.cross_entropy(logits_ba, aranged_gen_labels))
    return loss

def multimodal_pairwise_clip_loss(embeds: dict, temperature=0.07, weights: dict | None = None):
    """
    embeds: dict name -> (B,D)
    weights: optional dict with keys like ('alff','rsdata') or frozenset({'a','b'}) -> float
    """
    names = list(embeds.keys())
    losses = []
    wsum = 0.0

    for a, b in combinations(names, 2):
        w = 1.0
        if weights is not None:
            w = weights.get((a, b), weights.get((b, a), weights.get(frozenset([a, b]), 1.0)))
        losses.append(w * clip_loss_pair(embeds[a], embeds[b], temperature))
        wsum += w

    return sum(losses) / max(wsum, 1e-8)

def multimodal_multipositive_infonce_whole(embeds: dict, temperature=0.07):
    """
    embeds: dict name -> (B,D)
    For each modality as anchor, positives are other modalities same subject.
    ** Define this for comparison if this will be necessary - and check difference
    between average of pairs or whole loss**
    """
    names = list(embeds.keys())
    Z = [l2norm(embeds[n]) for n in names]  # each (B,D)
    M = len(Z)
    B, D = Z[0].shape

    # stack to (M,B,D)
    Z = torch.stack(Z, dim=0)

    # We'll compute for each anchor modality i:
    # logits over all (modality j, sample k) pairs: shape (B, M*B)
    # positives are positions where j != i and k == b (same subject)
    total_loss = 0.0
    count = 0

    for i in range(M):
        anchor = Z[i]  # (B,D)

        # (B,D) @ (M*B,D)^T -> (B, M*B)
        others = Z.reshape(M * B, D)                        # (M*B,D)
        logits = (anchor @ others.T) / temperature          # (B,M*B)

        # build positive mask: (B, M*B)
        # index mapping: flat index = j*B + k
        pos_mask = torch.zeros((B, M * B), device=anchor.device, dtype=torch.bool)
        for j in range(M):
            if j == i:
                continue
            idxs = j * B + torch.arange(B, device=anchor.device)
            pos_mask[torch.arange(B, device=anchor.device), idxs] = True

        # logsumexp over all candidates - as a sofmax normalization and regularize metrics across batch
        log_den = torch.logsumexp(logits, dim=1)  # (B,)

        # logsumexp over positives (multi-positive)
        # set non-positives to -inf
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(logits_pos, dim=1)  # (B,)

        # define the whole loss here den - num in the logarithmic domain
        loss_i = (log_den - log_num).mean()
        total_loss += loss_i
        count += 1

    return total_loss / count
