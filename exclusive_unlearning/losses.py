"""Forget and retain objectives.

All forget losses take `logits` (B, T, V) and `labels` (B, T) with IGNORE_INDEX
masking, and return a scalar averaged over supervised positions.

  ga          : -CE  (gradient ascent on the forget set; unbounded)
  npo         : -(2/beta) * mean log sigmoid(-beta * (log pi_theta - log pi_ref))
  max_entropy : KL( Uniform(V) || pi_theta(.|x) ) = log|V| - mean_v log pi_theta(v|x)
  retain      : CE on the retain set
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX


def _shift(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Next-token alignment: logits[:, :-1] predicts labels[:, 1:]. Returns (logits, labels, mask)."""
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    mask = labels != IGNORE_INDEX
    return logits, labels, mask


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence sum of log pi(y_t | y_<t) over supervised positions. Shape (B,)."""
    logits, labels, mask = _shift(logits, labels)
    logp = torch.log_softmax(logits, dim=-1)
    tok = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(dim=-1)


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits, labels, mask = _shift(logits, labels)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )


def retain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return cross_entropy(logits, labels)


def gradient_ascent_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -cross_entropy(logits, labels)


def npo_loss(
    logits: torch.Tensor, labels: torch.Tensor, ref_logits: torch.Tensor, beta: float = 0.1
) -> torch.Tensor:
    """Negative Preference Optimization (Zhang et al., 2024).

    L = -(2/beta) * E[ log sigmoid( -beta * (log pi_theta(y|x) - log pi_ref(y|x)) ) ]
    Bounded below; gradient weight vanishes as pi_theta(y|x) -> 0.
    """
    lp = sequence_logprobs(logits, labels)
    with torch.no_grad():
        lp_ref = sequence_logprobs(ref_logits, labels)
    _, _, mask = _shift(logits, labels)
    n = mask.sum(dim=-1).clamp_min(1).float()
    # Length-normalise the log-ratio so beta is comparable across sequence lengths.
    log_ratio = (lp - lp_ref) / n
    return -(2.0 / beta) * F.logsigmoid(-beta * log_ratio).mean()


def max_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """KL( U(V) || pi_theta ) averaged over supervised positions.

    KL(U||p) = sum_v (1/V) log( (1/V) / p_v ) = -log V - (1/V) sum_v log p_v.
    Minimum 0 at p = U; no divergence as in GA.
    """
    logits, labels, mask = _shift(logits, labels)
    logp = torch.log_softmax(logits / temperature, dim=-1)
    V = logp.size(-1)
    kl = -math.log(V) - logp.mean(dim=-1)  # (B, T)
    return (kl * mask).sum() / mask.sum().clamp_min(1)


def token_entropy(logits: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    """Mean predictive entropy (nats) over supervised positions (or all positions)."""
    if labels is None:
        lg = logits[:, :-1, :].float()
        mask = torch.ones(lg.shape[:2], dtype=torch.bool, device=lg.device)
    else:
        lg, _, mask = _shift(logits, labels)
    logp = torch.log_softmax(lg, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    return (ent * mask).sum() / mask.sum().clamp_min(1)


def forget_loss(
    objective: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    ref_logits: torch.Tensor | None = None,
    npo_beta: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    if objective == "ga":
        return gradient_ascent_loss(logits, labels)
    if objective == "npo":
        if ref_logits is None:
            raise ValueError("npo requires ref_logits from the frozen reference model")
        return npo_loss(logits, labels, ref_logits, beta=npo_beta)
    if objective == "max_entropy":
        return max_entropy_loss(logits, labels, temperature=temperature)
    raise ValueError(f"unknown forget objective {objective!r}")
