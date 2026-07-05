"""
quantum_models/quantum_aerial_adapter.py

Aerial-Selective Quantum Adapter (ASQA).

Applies VQC residual correction ONLY to aerial-camera tracklets (C4, C5 in AG-VPReID).
Ground-camera tracklets pass through unchanged.

    aerial_mask  = 1.0 if cam ∈ {4, 5} else 0.0   — hard binary gate (not learned)
    delta        = upscale(VQC(pre_net(x)))          — quantum (or classical) correction
    output       = x + aerial_mask * delta

Motivation: aerial cameras (70–120m altitude) produce degraded, lower-quality CLIP features.
Literature shows VQC advantage on noisy/constrained-dimensionality inputs (satellite imagery,
medical imaging). Ground cameras produce high-quality features where classical suffices.

Unlike CCG (camera-conditioned learned gate), ASQA uses a hard deterministic mask from the
camera label. No gate_net parameters — routing is fixed by camera ID at construction time.

bypass_quantum=True: delta from classical Linear+ReLU instead of VQC (mask still applied).
Ablation: does the benefit come from the quantum circuit, or from the selective correction?

cam_label=None fallback: passes x through unchanged (no routing info available).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml

# Aerial camera IDs for AG-VPReID (C4 and C5 are 70-120m altitude drone cameras)
AERIAL_CAMS = {4, 5}


class AerialSelectiveAdapter(nn.Module):
    """
    Hard-masked quantum residual adapter — VQC correction only for aerial tracklets.

    output = x + aerial_mask * upscale(VQC(pre_net(x)))

    aerial_mask[i] = 1.0 if cam_label[i] ∈ {4, 5} else 0.0

    Args:
        in_features    (int): Feature dimension (e.g. 768).
        n_qubits       (int): Number of qubits. Default 8.
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        bypass_quantum (bool): Replace VQC with Linear+ReLU for ablation.
        device_name    (str): PennyLane device. Default 'default.qubit'.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_quantum_features = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Pre-net: compress in_features → n_qubits angles.
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if bypass_quantum:
            self.classical_expansion = nn.Sequential(
                nn.Linear(n_qubits, self.n_quantum_features, bias=False),
                nn.ReLU(),
            )
        else:
            dev = qml.device(device_name, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # Upscale: 2^n_qubits → in_features. Near-zero init → delta ≈ 0 at init.
        self.upscale = nn.Linear(self.n_quantum_features, in_features, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if self.bypass_quantum:
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def forward(self, x: torch.Tensor, cam_label=None):
        """
        Args:
            x         : [B, in_features]
            cam_label : [B] camera IDs (LongTensor or None).
                        If None, returns x unchanged (no routing info).
        Returns:
            x_adapted [B, in_features]
        """
        if cam_label is None:
            return x

        input_dtype  = x.dtype
        x_f = x.float()

        # Hard binary aerial mask: [B, 1]
        aerial_mask = torch.tensor(
            [1.0 if int(c.item()) in AERIAL_CAMS else 0.0 for c in cam_label],
            dtype=torch.float32,
            device=x.device,
        ).unsqueeze(1)  # [B, 1]

        # Early exit if no aerial tracklets in this batch (ground-only batch)
        if aerial_mask.sum() == 0:
            return x

        # Correction branch
        angles = torch.sigmoid(self.pre_net(x_f)) * math.pi   # [B, n_qubits]
        if self.bypass_quantum:
            q_feat = self.classical_expansion(angles)
        else:
            q_feat = self.qlayer(angles.float())
        delta = self.upscale(q_feat)   # [B, in_features]

        # Apply correction only to aerial tracklets
        x_adapted = (x_f + aerial_mask * delta).to(dtype=input_dtype)
        return x_adapted

    def extra_repr(self) -> str:
        mode = "classical_bypass" if self.bypass_quantum else "VQC"
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, aerial_cams={sorted(AERIAL_CAMS)}, mode={mode}"
        )
