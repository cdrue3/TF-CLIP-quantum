# Survey Paper: Toward Quantum-Enhanced Computer Vision
**Authors**: Connor Druett et al. (QUT, Monash, KIT)
**File**: `Survey_Paper Revised draft.pdf` (52 pages)
**Load with**: `conda run -n tfclip python -c "import fitz; doc=fitz.open('Survey_Paper Revised draft.pdf'); [print(p.get_text()) for p in doc]"`

## Core Taxonomy
2D taxonomy: **Integration Level** × **Integration Pattern**

Integration Levels: Preprocessing | Feature Extraction | Classification | Optimization | Postprocessing

Integration Patterns:
- **Sequential**: unidirectional flow (classical→quantum or quantum→classical)
- **Parallel**: both run concurrently, outputs fused
- **Interlaced**: bidirectional loops between classical and quantum

## Our TF-CLIP Architecture Classification
**Sequential (classical-first) + Classification level** = "Dressed Quantum Circuit"
- Large CLIP backbone → quantum classifier heads (pre_net→VQC→post_net)
- This is the most common pattern in literature

## Encoding Methods (Table 3) — Critical for our design
| Method | Features per N qubits | Notes |
|---|---|---|
| Angle | N | Simple, low circuit depth, HIGH qubit requirement |
| **Dense Angle** | **2N** | Uses phase + angle; `|ψ_j⟩ = cos(πx_{2j-1})|0⟩ + e^{i2πx_{2j}}sin(πx_{2j-1})|1⟩` |
| Amplitude | 2^N | Highly efficient but O(2^N) gate depth |
| Hamiltonian | N to 4N-1 | Operator coefficient encoding |

**Dense Angle is directly applicable**: doubles information (16 values → 8 qubits) with same circuit depth as angle encoding. Phase info participates in interference after entangling gates.

## Data Re-uploading (Section 3.2.1)
"By repeatedly encoding classical input into a single or few-qubit system between processing layers, the model can approximate complex non-linear functions of a higher degree than would be possible with a single upload."
- Directly validates our planned data re-uploading approach
- Cited as [83] in survey: Pérez-Salinas et al. 2020
- Paper notes it's especially useful for NISQ devices to improve expressivity

## Key Problems Directly Applicable to TF-CLIP
1. **Multi-class degradation** (Section 4.1): "hybrid model attaining ~90% on MNIST binary drops to ~30% on 10-class MNIST." Our case: 625 classes. This is a fundamental known limitation.

2. **Dimensionality compression bottleneck** (Section 3.3.1): "Sequential architectures... the dimensionality compression causes accuracy degradation in multi-class tasks." Our 768→8 bottleneck is exactly this.

3. **Classical overshadowing** (Section 3.3.1): "Overly expressive classical pre-/post-processing layers have also been seen to potentially overshadow quantum components." Directly explains why our classical ablation beats the VQC.

4. **Barren plateau** (Section 5): "As qubit count/circuit depth increases, gradients vanish." More complex ansatz (StronglyEntanglingLayers) is WORSE — confirmed by our experiments.

## Recommended Approaches from Survey (not yet tried)

### Dense Angle Encoding (highest priority)
- Encode 2 features per qubit instead of 1 using phase
- pre_net: Linear(768, 2*n_qubits) → split into angles + phases
- Circuit: `RY(angle_j*π)` + `PhaseShift(phase_j*2π)` per qubit (or qml.IQPEmbedding)
- Doubles information flowing into VQC without changing n_qubits
- No null gradient concern (single embedding, angles still near π/2 at init)
- In PennyLane: use `qml.RY` + `qml.RZ` (phase) per qubit, then entanglement

### Interlaced + Data Re-uploading (Section 3.3.3 + 5)
"Further research into interlaced designs, especially with data re-uploading, offers a way to capture complex information in visual data without exceeding current qubit limitations."
- Multiple independent pre_nets (one per layer), interleaved embedding
- Must use std=0.2 VQC init (not 0.01) to avoid null gradient for re-uploading

### Parallel Pattern (Section 3.3.2 + future directions)
"Parallel networks offer resilience against quantum noise and improved performance."
- Run classical and quantum classifiers in parallel, fuse outputs
- Primacy issue: LR scheduling needed to prevent classical stream from dominating
- But: paper recommends this as future direction; simpler to implement

## Integration Pattern Trade-offs (Table 5)
| Pattern | When to Choose | Key Challenge |
|---|---|---|
| Sequential | Low complexity, NISQ feasibility | Classical layers overshadow quantum |
| Parallel | Global+local patterns, noise robustness | Primacy issue, LR scheduling |
| Interlaced | Iterative refinement, multi-stage | Latency, complexity, fragility |

## Relevant Performance Results (Table 6)
- Sequential classifier: Rice leaf 99.93%; Abdominal trauma AUC 0.77 (+0.14)
- Parallel: CIFAR-10 71.6% (+2.9%)
- Interlaced Q-C-Q: Brain tumor 95.18% (+5.7% over classical)
- All with 4-10 qubits; all note performance degrades with class count

## Conclusion Relevant Quotes
- "Hybrid architectures do not yet offer holistic superiority over classical models, they present a promising pathway for leveraging quantum advantages within practical constraints"
- "Future research should focus on resource-aware designs, efficient encoding strategies, and exploration of parallel and interlaced patterns"
- Main finding: hybrid models excel in sample efficiency and data-constrained scenarios, not necessarily in all benchmarks
