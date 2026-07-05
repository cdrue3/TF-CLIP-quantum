"""
quantum_models/angle/quantum_amplitude_temporal.py

Quantum Amplitude Temporal (QAT) — QPCA + Quantum Transfer Learning.

Replaces classical mean-pool with a quantum circuit that amplitude-encodes
the FULL 768-dim ViT feature vector into a 10-qubit quantum state (2^10=1024
amplitudes ≥ 768), then applies a parameterised unitary (StronglyEntanglingLayers)
as the quantum PCA rotation.

No pre_net squashing: all 768 features are preserved as quantum state amplitudes.
The VQC rotation is the NISQ analogue of quantum PCA — a learnable unitary
diagonalising the feature density matrix in the basis most useful for ReID.

Architecture:
    Input x: [B, T, in_features]
    1. mean_pool(x):  [B, T, 768] → [B, 768]              (classical aggregation)
    2. AmplitudeEmbedding(mean_feat, pad_with=0, normalize=True)
                      [B, 768] → 10-qubit state            (pad 768→1024=2^10)
    3. StronglyEntanglingLayers(weights, wires=range(10))  (quantum PCA rotation)
    4. qml.probs → [B, 1024]                               (all 2^10 outcomes)
    5. upscale: Linear(1024 → 768, bias=False)             (init N(0, 0.001))
    6. Skip: output = mean_pool(x) + delta                 (near-zero at init)
    Output: [B, 768]

bypass_quantum=True: returns x.mean(1) directly (classical mean-pool ablation).
"""

import torch
import torch.nn as nn
import pennylane as qml


class QuantumAmplitudeTemporal(nn.Module):
    """
    Quantum Amplitude Temporal Aggregation: [B, T, in_features] → [B, in_features].

    Full-fidelity quantum encoding: 768 ViT features are amplitude-encoded into
    a 10-qubit state (2^10=1024 amplitudes), with no pre-compression bottleneck.
    The VQC applies a parameterised unitary rotation — the quantum PCA step.

    Skip connection: output = mean_pool(x) + upscale(VQC(amplitude_encode(mean_pool(x)))).
    At init upscale is near-zero → output ≈ mean_pool(x).

    Args:
        in_features    (int): Feature dimension. Default 768 (ViT-B/16).
        n_qubits       (int): Qubit count. Must satisfy 2^n_qubits >= in_features.
                              Default 10 → 2^10=1024 ≥ 768.
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        bypass_quantum (bool): If True, skip VQC and return plain mean-pool.
        device_name    (str): PennyLane device. Default 'default.qubit'.
    """

    def __init__(
        self,
        in_features: int = 768,
        n_qubits: int = 10,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.n_amplitudes   = 2 ** n_qubits   # 1024 for n_qubits=10
        self.bypass_quantum = bypass_quantum

        assert self.n_amplitudes >= in_features, (
            f"2^n_qubits={self.n_amplitudes} must be >= in_features={in_features}. "
            f"Use n_qubits >= {in_features.bit_length()} (got {n_qubits})."
        )

        if not bypass_quantum:
            dev = qml.device(device_name, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(features, weights):
                # features: [B, in_features] — padded to n_amplitudes, normalised
                # weights:  [n_layers, n_qubits, 3]
                # AmplitudeEmbedding pads 768→1024 with zeros and normalises.
                qml.AmplitudeEmbedding(
                    features, wires=range(n_qubits),
                    normalize=True, pad_with=0.
                )
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            self.circuit = _circuit

            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_qubits
            )
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        # Upscale: 2^n_qubits → in_features. Near-zero init → skip starts as mean-pool.
        self.upscale = nn.Linear(self.n_amplitudes, in_features, bias=False)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

        if not bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, in_features]  (float16 or float32, any device)
        Returns:
            [B, in_features]
        """
        mean_feat = x.mean(1)   # [B, 768] — always computed for skip

        if self.bypass_quantum:
            return mean_feat

        input_dtype = x.dtype

        # Amplitude-encode full 768-dim feature. AmplitudeEmbedding:
        #   - pads [B, 768] → [B, 1024] with zeros
        #   - normalises each row to unit norm
        features = mean_feat.float()   # [B, 768]

        q_out = self.circuit(features, self.qlayer_weights).float()  # [B, 1024]

        delta = self.upscale(q_out)   # [B, 768]

        return (mean_feat.float() + delta).to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_amplitudes={self.n_amplitudes}, n_layers={self.n_layers}, "
            f"bypass={self.bypass_quantum}"
        )
