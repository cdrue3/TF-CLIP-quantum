import math
import torch
import torch.nn as nn
import pennylane as qml


class QTDAmp(nn.Module):
    """
    QTD with Amplitude Encoding — compression-free alternative to angle encoding.

    Replaces the 768→n_qubits pre_net bottleneck with full-fidelity amplitude
    embedding. Each frame difference is padded to 2^n_qubits and normalised, then
    embedded directly as quantum state amplitudes. No learnable encoding layer.

    n_qubits is derived from in_features: ceil(log2(in_features)).
    For in_features=768: n_qubits=10, pad_to=1024.

    Each of the T-1 differences is run through the shared circuit independently;
    outputs are averaged, then residual on mean_pool.
    AmplitudeEmbedding resets the full state per call — re-uploading is NOT used
    (it would overwrite previous differences). Processing is independent per diff.

    Caveat: normalisation discards L2 magnitude. Near-zero diffs (slow motion) are
    clamped to 1e-8 to avoid NaN. The skip connection preserves original magnitude.
    """

    def __init__(self, in_features=768, n_layers=2, seq_len=8,
                 bypass_quantum=False, device_name='default.qubit'):
        super().__init__()
        self.in_features    = in_features
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.n_diffs        = seq_len - 1
        self.bypass_quantum = bypass_quantum

        self.n_qubits       = math.ceil(math.log2(in_features))  # 10 for 768
        self.pad_to         = 2 ** self.n_qubits                  # 1024
        self.n_measurements = self.pad_to

        if not bypass_quantum:
            n_q = self.n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def _circuit(features, weights):
                # features: [B, pad_to] — normalised real amplitudes (pre-normalised)
                # weights:  [n_layers, n_q, 3]
                qml.AmplitudeEmbedding(features, wires=range(n_q), normalize=False)
                qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_q
            )
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        self.upscale = nn.Linear(self.n_measurements, in_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def _apply(self, fn):
        super()._apply(fn)
        if not self.bypass_quantum:
            self.qlayer_weights.data = self.qlayer_weights.data.cpu().float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_features] → [B, in_features]"""
        mean_feat = x.mean(1)
        if self.bypass_quantum:
            return mean_feat

        input_dtype  = x.dtype
        input_device = x.device
        B, T, D = x.shape

        diffs = x[:, 1:] - x[:, :-1]                                   # [B, T-1, D]
        diffs_flat = diffs.float().reshape(B * self.n_diffs, D)         # [B*(T-1), D]

        # Pad to 2^n_qubits
        if self.pad_to > D:
            pad = torch.zeros(B * self.n_diffs, self.pad_to - D,
                              device=diffs_flat.device, dtype=torch.float32)
            diffs_flat = torch.cat([diffs_flat, pad], dim=1)            # [B*(T-1), 1024]

        # Normalise for AmplitudeEmbedding (||x||=1 required)
        norms       = diffs_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        diffs_normed = (diffs_flat / norms).cpu()                       # [B*(T-1), 1024]

        q_out = self.circuit(
            diffs_normed,
            self.qlayer_weights.cpu().float()
        ).float()                                                        # [B*(T-1), 1024]

        q_out = q_out.reshape(B, self.n_diffs, self.n_measurements).mean(1)  # [B, 1024]

        delta = self.upscale(q_out.to(input_device))
        return (mean_feat.float() + delta).to(input_dtype)

    def extra_repr(self):
        return (f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
                f"pad_to={self.pad_to}, n_layers={self.n_layers}, "
                f"n_diffs={self.n_diffs}, bypass={self.bypass_quantum}")
