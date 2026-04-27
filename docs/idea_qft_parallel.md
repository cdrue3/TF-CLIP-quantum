# Idea: QFT Parallel Branch in VQC

## Concept
Replace or augment `StronglyEntanglingLayers` with a parallel QFT branch inside the VQC.

**Motivation**: QFT as a structured, parameter-free unitary transforms the angle-encoded feature representation into the frequency basis. Different identities may have distinct spectral signatures in the compressed 8-qubit space. Aerial/ground viewpoint gap might be more invariant in frequency domain than raw feature domain.

## Proposed Architecture

```
AngleEmbedding(angles)
       |
  _____|_____
 |           |
QFT        StronglyEntanglingLayers(weights)  ← learnable
 |           |
measure    measure
[256]      [256]
  |___________|
      concat or add
          |
     upscale [→768]
          |
      residual on mean_pool
```

- QFT branch: zero learnable circuit parameters, no barren plateau risk, structured frequency mixing
- SEL branch: retains learnable expressivity, adapts to re-ID task
- Combined: geometric interpretation + adaptability

## Why QFT alone feels incomplete
Zero learnable quantum parameters means the circuit can't adapt to what "similar" means for person re-ID. The parallel design lets QFT contribute structure while SEL handles task-specific learning.

## Why this is novel
QFT is used in Shor's, phase estimation etc — not as a feature mixing layer in hybrid QML for vision. Parallel QFT+SEL in a re-ID VQC is unpublished framing, worth a paragraph in the WACV paper.

## Implementation notes
- `qml.QFT(wires=range(n_q))` is built into PennyLane
- O(n²) gates for n qubits — 64 gates at n=8, easily simulatable
- Concat path: upscale input doubles to [512] → Linear(512, 768)
- Add path: both branches measure [256], element-wise add, upscale [256→768] — simpler, fewer parameters

## Priority
Low — revisit once Colab/Drive setup is running and baseline comparison runs are done.
