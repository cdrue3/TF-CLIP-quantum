"""
quantum_models/quantum_gated_adapter_ccg.py

Camera-Conditioned Gated Quantum Adapter (CCG).

Extends GatedQuantumAdapter by conditioning the routing gate on camera ID:

    cam_e  = cam_gate_embed(cam_label)          [B, cam_embed_dim]
    g(x)   = sigmoid(gate_net(concat(x, cam_e)))  ∈ (0, 1)
    delta  = upscale(VQC(pre_net(x)))
    output = x + g * delta

AG-ReID has 2 cameras:
    cam 0 = ground (standard side-view)
    cam 1 = aerial (top-down drone view)

These produce systematically different CLIP feature distributions. The implicit
gate (GatedQuantumAdapter) may not separate cameras if the camera signal in the
768-dim features is too diluted. CCG provides an explicit 16-dim camera channel
so the gate can learn camera-specific routing more easily.

bypass_quantum=True: delta from classical Linear+ReLU (gate still camera-conditioned).
cam_label=None fallback: treats cam_label as 0 (safe for datasets without camera info).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class CameraConditionedGatedAdapter(nn.Module):
    """
    Camera-conditioned gated quantum residual adapter.

    output = x + sigmoid(gate_net(concat(x, cam_embed(cam_label)))) * upscale(VQC(pre_net(x)))

    Args:
        in_features    (int): Feature dimension (e.g. 768).
        n_qubits       (int): Number of qubits. Default 8.
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        bypass_quantum (bool): Replace VQC with Linear+ReLU for ablation.
        camera_num     (int): Number of camera IDs. Default 2.
        cam_embed_dim  (int): Dimension of camera embedding fed to gate. Default 16.
        device_name    (str): PennyLane device. Default 'default.qubit'.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        camera_num: int = 2,
        cam_embed_dim: int = 16,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features   = in_features
        self.n_qubits      = n_qubits
        self.n_layers      = n_layers
        self.n_quantum_features = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum
        self.cam_embed_dim = cam_embed_dim

        # Camera embedding for gate conditioning
        self.cam_gate_embed = nn.Embedding(camera_num, cam_embed_dim)

        # Gate: conditioned on (x, cam_embed). bias=0 → g=0.5 at init (neutral).
        self.gate_net = nn.Linear(in_features + cam_embed_dim, 1, bias=True)

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
        nn.init.zeros_(self.gate_net.bias)          # g = 0.5 at init
        nn.init.kaiming_normal_(self.gate_net.weight, a=0, mode="fan_in")
        nn.init.normal_(self.cam_gate_embed.weight, mean=0, std=0.01)
        if self.bypass_quantum:
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def to(self, *args, **kwargs):
        """Pin qlayer to CPU; all other modules follow device migrations."""
        super().to(*args, **kwargs)
        if not self.bypass_quantum:
            self.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)
        return self

    def forward(
        self, x: torch.Tensor, cam_label=None, return_gates: bool = False
    ):
        """
        Args:
            x            : [B, in_features]
            cam_label    : [B] camera IDs (LongTensor). If None, treated as cam 0.
            return_gates : If True, return (output, g) where g is [B] gate values.
        Returns:
            x_adapted [B, in_features], or (x_adapted, g [B]) if return_gates=True.
        """
        input_dtype  = x.dtype
        input_device = x.device
        x_f = x.float()
        B   = x_f.shape[0]

        # Camera embedding for gate conditioning
        if cam_label is None:
            cam_idx = torch.zeros(B, dtype=torch.long, device=input_device)
        else:
            cam_idx = cam_label.long().to(input_device)
        cam_e = self.cam_gate_embed(cam_idx).float()   # [B, cam_embed_dim]

        # Camera-conditioned gate: [B, 1]
        gate_input = torch.cat([x_f, cam_e], dim=1)   # [B, in_features + cam_embed_dim]
        g = torch.sigmoid(self.gate_net(gate_input))   # [B, 1]

        # Correction branch (same as GatedQuantumAdapter)
        angles = torch.sigmoid(self.pre_net(x_f)) * math.pi   # [B, n_qubits]
        if self.bypass_quantum:
            q_feat = self.classical_expansion(angles)
        else:
            angles_cpu = angles.float()
            q_feat = self.qlayer(angles_cpu).to(input_device)
        delta = self.upscale(q_feat)   # [B, in_features]

        # Gated residual
        x_adapted = (x_f + g * delta).to(input_dtype)

        if return_gates:
            return x_adapted, g.squeeze(1).detach().cpu()
        return x_adapted

    def extra_repr(self) -> str:
        mode = "classical_bypass" if self.bypass_quantum else "VQC"
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, cam_embed_dim={self.cam_embed_dim}, mode={mode}"
        )
