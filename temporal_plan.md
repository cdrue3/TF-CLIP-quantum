# Quantum Temporal Variants — Future Work

## Already Implemented
- **TQA (QTemporal)**: data re-uploading over T frames sequentially in shared VQC. Mean-pool skip connection.

---

## Candidate Architectures

### 1. Quantum Temporal Difference (QTD)
Encode *motion* rather than absolute frame content.
Feed frame differences `Δ_t = frame_{t+1} - frame_t` (T-1 difference vectors) through VQC.
```
[Δ_1, Δ_2, ..., Δ_{T-1}] → pre_net(768→n_q) → VQC (batched) → probs → upscale → residual on mean_pool
```
- Cheapest to implement (same circuit as TQA, different input)
- Clear motivation: motion cues ignored by mean-pool and TQA alike
- T-1 = 3 differences for seq_len=4

### 2. Quantum Frame Correlation (QFC)
Process pairs of frames jointly — measure quantum correlations *between* frames.
```
[frame_i || frame_j] → pre_net(2×768→n_q) → VQC → probs → average over pairs → upscale → residual
```
- More expensive: T(T-1)/2 pairs per sample
- Genuinely captures cross-frame quantum interference (impossible classically with same architecture)
- For seq_len=4: 6 pairs

### 3. Quantum-Gated Temporal (QGT)
Gate controls how much of TQA correction to apply, conditioned on mean-pooled tracklet.
Directly answers KIT Research Q2 ("which inputs benefit from quantum?") for the temporal domain.
```
g = sigmoid(gate_net(mean_pool(x)))       ← scalar: is this tracklet temporally complex?
output = mean_pool(x) + g * TQA_delta(x)
```
- Minimal code change on top of TQA
- Interpretable: log g values to see which tracklets benefit from temporal quantum correction
- Aerial vs ground gate behaviour would be interesting to analyse

---

## Priority Order
1. QTD — fastest to implement, strong motion motivation
2. QGT — builds on TQA, interpretable gate analysis
3. QFC — most novel but more expensive
