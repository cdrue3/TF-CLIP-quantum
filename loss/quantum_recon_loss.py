"""
loss/quantum_recon_loss.py

Reconstruction loss component for QuantumAutoEncoder.

Adds MSE(x_recon, x_original) * recon_weight to the total training loss.
This forces the quantum bottleneck to preserve input information.

Used by train_qautoencoder.py alongside the standard ID + triplet + I2T losses.
"""

import torch
import torch.nn.functional as F


def quantum_recon_loss(x_recon: torch.Tensor, x_original: torch.Tensor,
                       recon_weight: float = 0.1) -> torch.Tensor:
    """
    MSE reconstruction loss between autoencoder output and original features.

    Args:
        x_recon     : [B, D] reconstructed features from quantum decoder
        x_original  : [B, D] original input features (detached — no grad through original)
        recon_weight: scalar weight on reconstruction loss (default 0.1)

    Returns:
        scalar loss tensor
    """
    return recon_weight * F.mse_loss(x_recon, x_original.detach())
