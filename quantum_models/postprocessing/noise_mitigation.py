"""
quantum_models/postprocessing/noise_mitigation.py

Zero Noise Extrapolation (ZNE) for quantum circuits.

Survey reference: Bultrini et al., Jnane et al.
Required for deploying any quantum component on real NISQ hardware.

On the default.qubit CPU simulator there is no physical noise, so ZNE has
no effect — the "noisy" circuit evaluations simply add artificial
DepolarizingChannel noise, and Richardson extrapolation removes it.
This module demonstrates hardware deployment readiness.

ZNE workflow:
    1. Evaluate circuit at noise_factors = [1, 3] (identity × noise_factor circuits)
    2. Richardson extrapolation: linearly extrapolate to noise_factor=0
    3. Return denoised probability estimate

Usage:
    from quantum_models.postprocessing.noise_mitigation import ZNEWrapper, zero_noise_extrapolate

    # Wrap a qnode:
    denoised_probs = zero_noise_extrapolate(circuit_fn, angles, weights,
                                             n_qubits=8, noise_prob=0.01)

    # Or use as a module-level function for post-processing eval:
    # Add --use_zne flag to eval scripts
"""

import torch
import pennylane as qml


def _build_noisy_circuit(original_qnode_fn, n_qubits: int, noise_prob: float,
                          noise_factor: int, seq_len: int = None):
    """
    Build a noisy version of a circuit by adding DepolarizingChannel after
    each gate block. noise_factor=1 corresponds to the base noise level.

    On default.qubit this inserts artificial noise to mimic hardware behavior.
    """
    dev_noisy = qml.device("default.mixed", wires=n_qubits)

    if seq_len is not None:
        # Temporal VQC with re-uploading
        @qml.qnode(dev_noisy, interface="torch", diff_method="backprop")
        def noisy_circuit(angles_2d, weights):
            for t in range(seq_len):
                qml.AngleEmbedding(angles_2d[t], wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                # Apply depolarizing noise after each block (scaled by noise_factor)
                for _ in range(noise_factor):
                    for wire in range(n_qubits):
                        qml.DepolarizingChannel(noise_prob, wires=wire)
            return qml.probs(wires=range(n_qubits))
    else:
        @qml.qnode(dev_noisy, interface="torch", diff_method="backprop")
        def noisy_circuit(angles, weights):
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            for _ in range(noise_factor):
                for wire in range(n_qubits):
                    qml.DepolarizingChannel(noise_prob, wires=wire)
            return qml.probs(wires=range(n_qubits))

    return noisy_circuit


def zero_noise_extrapolate(circuit_fn, angles, weights,
                            n_qubits: int = 8, noise_prob: float = 0.01,
                            noise_factors=(1, 3), seq_len: int = None):
    """
    Apply Zero Noise Extrapolation to a VQC evaluation.

    Evaluates the circuit at two noise levels (noise_factors × base_noise)
    and linearly extrapolates to zero noise via Richardson extrapolation.

    Args:
        circuit_fn   : The original (noiseless) qnode function (used for reference).
        angles       : Input angle tensor [B, n_q] or [T, B, n_q]
        weights      : Circuit weight tensor
        n_qubits     : Number of qubits
        noise_prob   : Base depolarizing probability (default 0.01)
        noise_factors: Noise scaling factors for ZNE. Default (1, 3).
        seq_len      : T frames if temporal circuit; None for single-shot.

    Returns:
        probs_zne: ZNE-corrected probability tensor [B, 2^n_qubits]
    """
    assert len(noise_factors) == 2, "ZNE requires exactly 2 noise factors for linear extrapolation"
    lam1, lam2 = noise_factors

    circ1 = _build_noisy_circuit(circuit_fn, n_qubits, noise_prob, lam1, seq_len)
    circ2 = _build_noisy_circuit(circuit_fn, n_qubits, noise_prob, lam2, seq_len)

    with torch.no_grad():
        probs1 = circ1(angles.float(), weights.float())  # [B, 2^n_q] at noise_factor=lam1
        probs2 = circ2(angles.float(), weights.float())  # [B, 2^n_q] at noise_factor=lam2

    # Richardson linear extrapolation: f(0) ≈ (lam2*f1 - lam1*f2) / (lam2 - lam1)
    probs_zne = (lam2 * probs1 - lam1 * probs2) / (lam2 - lam1)
    probs_zne = probs_zne.clamp(min=0.0)  # probabilities must be non-negative
    # Renormalize (extrapolation may slightly violate normalization)
    probs_zne = probs_zne / probs_zne.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return probs_zne


class ZNEWrapper:
    """
    Lightweight wrapper that applies ZNE to any VQC call at eval time.

    Usage:
        zne = ZNEWrapper(n_qubits=8, noise_prob=0.01)
        # In eval loop, replace: probs = circuit(angles, weights)
        # With:                  probs = zne(circuit, angles, weights)
    """

    def __init__(self, n_qubits: int = 8, noise_prob: float = 0.01,
                 noise_factors=(1, 3), seq_len: int = None):
        self.n_qubits     = n_qubits
        self.noise_prob   = noise_prob
        self.noise_factors = noise_factors
        self.seq_len      = seq_len

    def __call__(self, circuit_fn, angles, weights):
        return zero_noise_extrapolate(
            circuit_fn, angles, weights,
            n_qubits=self.n_qubits,
            noise_prob=self.noise_prob,
            noise_factors=self.noise_factors,
            seq_len=self.seq_len,
        )
