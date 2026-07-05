"""
model/feature_preprocessors.py

Quantum-inspired preprocessing modules that operate on [B, T, D] frame features
before temporal pooling. Drop-in replacements for img_feature.mean(1).

All modes return [B, D] — no downstream changes needed.

Modes:
  none   — plain mean pool (baseline, no-op)
  edge   — mean pool + mean of frame differences (adds motion signal)
  dft    — learnable frequency-weighted pooling via DFT across T
  pca    — learned whitening of mean-pooled features (identity init)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePreprocessor(nn.Module):
    def __init__(self, mode: str, in_features: int = 768, seq_len: int = 8):
        super().__init__()
        self.mode = mode
        self.in_features = in_features
        self.seq_len = seq_len

        if mode == 'pca':
            # Learned whitening — initialised as identity so baseline is unchanged
            self.whiten = nn.Linear(in_features, in_features, bias=True)
            nn.init.eye_(self.whiten.weight)
            nn.init.zeros_(self.whiten.bias)

        elif mode == 'dft':
            # Learnable per-frequency weights across T//2+1 DFT bins
            n_freqs = seq_len // 2 + 1  # 5 for T=8
            self.freq_weights = nn.Parameter(torch.zeros(n_freqs))
            # Init: concentrate on DC (freq 0) so baseline ≈ mean pool
            with torch.no_grad():
                self.freq_weights[0] = 1.0

        print(f"[FeaturePreprocessor] mode={mode}, in_features={in_features}, seq_len={seq_len}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, D]
        """
        if self.mode == 'none':
            return x.mean(1)

        elif self.mode == 'edge':
            # Mean pool + mean of consecutive frame differences
            # Adds motion/change signal to static appearance descriptor
            diffs = x[:, 1:] - x[:, :-1]          # [B, T-1, D]
            return x.mean(1) + diffs.mean(1)        # [B, D]

        elif self.mode == 'dft':
            # DFT across T dimension — captures temporal frequency content
            # freq 0 = DC = mean pool; higher freqs = motion patterns
            fft  = torch.fft.rfft(x.float(), dim=1)  # [B, T//2+1, D] complex
            mag  = torch.abs(fft)                     # [B, T//2+1, D] real
            w    = F.softmax(self.freq_weights, dim=0)# [T//2+1] — learnable blend
            out  = (mag * w.view(1, -1, 1)).sum(1)    # [B, D]
            return out.to(x.dtype)

        elif self.mode == 'pca':
            # Learned whitening on mean-pooled features
            return self.whiten(x.mean(1).float()).to(x.dtype)

        elif self.mode == 'phase':
            # Sine Fourier feature map before mean pooling
            return torch.sin(x.float() * math.pi).mean(1).to(x.dtype)

        else:
            raise ValueError(f"Unknown preprocess mode: {self.mode}")

    def extra_repr(self):
        return f"mode={self.mode}, in_features={self.in_features}, seq_len={self.seq_len}"
