"""
Robust loss functions for training with noisy labels.

Contains:
- SymmetricCrossEntropy (Wang et al., 2019)
- ForwardCorrectionLoss (Patrini et al., 2017)
- estimate_transition_matrix: anchor-based T estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SymmetricCrossEntropy(nn.Module):
    """Symmetric Cross-Entropy loss (Wang et al., ICCV 2019).

    L_SCE = alpha * CE(p, q) + beta * CE(q, p)

    The reverse CE term is bounded and thus robust to noisy labels.
    """

    def __init__(self, alpha=1.0, beta=1.0, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        # Standard CE: -sum(p * log(q))
        ce = F.cross_entropy(logits, targets)

        # Reverse CE: -sum(q * log(p))
        one_hot = F.one_hot(targets, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        rce = -torch.sum(probs * torch.log(one_hot), dim=1).mean()

        return self.alpha * ce + self.beta * rce


class ForwardCorrectionLoss(nn.Module):
    """Forward loss correction (Patrini et al., CVPR 2017).

    Given an estimated noise transition matrix T where T_ij = P(y_noisy=j | y_true=i),
    the corrected prediction is: p_corrected = T^T @ softmax(logits).
    Loss = -log(p_corrected[noisy_label]).
    """

    def __init__(self, T):
        super().__init__()
        # T: (num_classes, num_classes) transition matrix
        self.register_buffer('T', T)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)  # (B, C)
        # Apply forward correction: T^T @ p
        corrected = probs @ self.T  # (B, C) @ (C, C) = (B, C)
        corrected = torch.clamp(corrected, min=1e-7)
        log_corrected = torch.log(corrected)
        loss = F.nll_loss(log_corrected, targets)
        return loss


def estimate_transition_matrix(model, features, noisy_labels, num_classes=10, device='cpu'):
    """Estimate the noise transition matrix T using the anchor-based approach.

    For each class i, find the training sample with the highest predicted probability
    for class i (the "anchor point"). Use that sample's predicted distribution as the
    i-th row of T: T[i, :] = softmax(logits_anchor).

    This assumes that for each class there exists at least one clean sample that the
    model is very confident about.

    Args:
        model: trained linear classifier
        features: (N, D) tensor of extracted features
        noisy_labels: (N,) tensor of noisy labels (unused for estimation, included for API)
        num_classes: number of classes
        device: torch device

    Returns:
        T: (num_classes, num_classes) estimated transition matrix
    """
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        probs = torch.softmax(logits, dim=1).cpu()  # (N, C)

    T = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        # Find the sample most confidently predicted as class i
        anchor_idx = probs[:, i].argmax().item()
        T[i, :] = probs[anchor_idx, :]

    # Normalize rows to sum to 1
    T = T / T.sum(dim=1, keepdim=True)
    return T
