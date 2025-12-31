"""
  Losses definition SSL
  and possibilities for regularization
"""

import torch
import torch.nn.functional as F
from itertools import combinations

def l2norm(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def clip_loss_pair(z_a, z_b, temperature=0.07):
    # z_a, z_b: (B,D)
    z_a = l2norm(z_a)
    z_b = l2norm(z_b)

    logits_ab = (z_a @ z_b.T) / temperature
    logits_ba = (z_b @ z_a.T) / temperature

    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss = 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels))
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
