"""
quantum_models/feature_extraction/quantum_temporal_sqp.py

Temporal Quantum Aggregation with Stochastic Quantum Perturbation (SQP).

Identical to QuantumTemporalDeep (n_layers=2 default) with two additions:

1. STOCHASTIC QUANTUM PERTURBATION (SQP):
   During training, Gaussian noise is injected into circuit weights before each
   forward pass: weights_noisy = weights + _noise_scale * N(0,1).
   This approximates noise-induced regularization (Kuzmin 2025, arXiv:2410.19921),
   which smooths the quantum loss landscape by suppressing high-frequency Fourier
   components. The noise scale is set externally and decayed to zero during training.

   Key property: straight-through gradient — noise is treated as a constant during
   backprop, so gradients flow to qlayer_weights unchanged. This is equivalent to
   variational inference over quantum circuit parameters.

2. BORN ENTROPY EXPOSURE:
   self._last_probs stores the VQC output probabilities after each forward pass.
   The training script uses these to compute Shannon entropy H(p) and adds
   -lambda * H(p) to the loss, maximising circuit entropy as soft regularisation.

   The entropy term prevents the circuit from collapsing to a low-entropy
   (over-deterministic) state at the ep30 LR drop, where fixed LR schedules
   cause the hardest adaptation pressure.

References:
  - Kuzmin et al. 2025, "Method for noise-induced regularization in QNNs"
  - Regularizing quantum loss landscapes by noise injection, Phys. Rev. A 2025
  - On the Expressibility and Overfitting of Quantum Circuit Learning, ACM TQC 2021
"""

import math
import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalSQP(nn.Module):
    """
    TQA + Stochastic Quantum Perturbation + Born Entropy Exposure.

    Identical to QuantumTemporalDeep except:
    - self._noise_scale (float): Gaussian noise std added to weights during training.
      Set externally each epoch: model.tqa._noise_scale = σ(epoch).
    - self._last_probs (Tensor or None): VQC output probs after last forward pass.
      Accessed by training script for entropy regularization.

    Args:
        in_features    (int): Feature dimension (768 for ViT-B-16).
        n_qubits       (int): Qubit count. Default 8 → 256 outcomes.
        n_layers       (int): VQC depth. Default 2.
        seq_len        (int): T — frames per tracklet. Default 8.
        bypass_quantum (bool): If True, return plain mean-pool.
        device_name    (str): PennyLane device.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        seq_len: int = 8,
        bypass_quantum: bool = False,
        dense_encoding: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum
        self.dense_encoding = dense_encoding

        # External control: set before each epoch
        self._noise_scale = 0.0
        self._last_probs  = None  # exposed for entropy loss in training script

        pre_net_out = 2 * n_qubits if dense_encoding else n_qubits
        self.pre_net = nn.Linear(in_features, pre_net_out, bias=False)

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            if dense_encoding:
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(angles_2d, weights):
                    for t in range(seq_len):
                        qml.AngleEmbedding(angles_2d[t, :, :n_q], wires=range(n_q), rotation="Y")
                        qml.AngleEmbedding(angles_2d[t, :, n_q:], wires=range(n_q), rotation="Z")
                        qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                    return qml.probs(wires=range(n_q))
            else:
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(angles_2d, weights):
                    for t in range(seq_len):
                        qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
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
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.005)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_feat = x.mean(1)
        if self.bypass_quantum:
            return mean_feat

        input_dtype = x.dtype
        B, T, D = x.shape

        pre_net_dim = 2 * self.n_qubits if self.dense_encoding else self.n_qubits
        angles = torch.sigmoid(self.pre_net(x.float().reshape(B * T, D))) * math.pi
        angles = angles.reshape(B, T, pre_net_dim)

        angles_f  = angles.permute(1, 0, 2).float()
        weights_f = self.qlayer_weights.float()

        # SQP: inject Gaussian noise into circuit weights during training
        if self.training and self._noise_scale > 0.0:
            weights_f = weights_f + self._noise_scale * torch.randn_like(weights_f)

        q_out = self.circuit(angles_f, weights_f).float()

        # Born entropy exposure: store probs for entropy regularisation in training script
        self._last_probs = q_out.detach()

        delta = self.upscale(q_out)
        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        vqc_params = self.n_layers * self.n_qubits * 3
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"vqc_params={vqc_params}, SQP+BER enabled, bypass={self.bypass_quantum}"
        )
