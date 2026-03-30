# Quantum TF-CLIP — Architecture Explainer

---

## Part 1: CLIP and TF-CLIP

### CLIP

CLIP (Contrastive Language-Image Pre-training, OpenAI 2021) is a model trained on 400 million
image-text pairs from the internet. The training task is simple: given a batch of (image, caption)
pairs, learn to match each image to its correct caption and push apart non-matching pairs. This is
done with a contrastive loss — no class labels, no bounding boxes, just "which image goes with
which text."

The result is two encoders trained to produce embeddings in a shared 512-dim semantic space:
- A visual encoder (ViT-B/16 in our case)
- A text encoder (a standard Transformer)

After training, cos_similarity(encode(image), encode("a photo of a cat")) is high if the image
contains a cat, without any category-specific fine-tuning.

---

### The ViT-B/16 visual encoder

ViT (Vision Transformer) treats an image like a sentence of patches:

    Image (256×128 px)
      → divide into 16×16 px non-overlapping patches → 128 patches
      → each patch flattened + linearly projected → 768-dim vector (the "patch token")
      → prepend a [CLS] learnable token → sequence of 129 tokens
      → add position embeddings (so the model knows where each patch is)
      → feed through 12 layers of standard Transformer (self-attention + MLP)
      → output: 129 vectors of 768 dim each
           token[0]  = [CLS] token = global image summary (this is what we use)
           tokens[1:] = per-patch local features

The [CLS] token attends to all 128 patches across all 12 layers, so it accumulates global
information. It ends up as a 768-dim vector summarising the whole image.

There is also a linear projection head on top of the [CLS] token that maps 768→512, which is the
dimension that lives in the shared image-text space. So the backbone gives two representations per
image:
- img_feature: 768-dim (raw ViT output, richer, used for ID classification)
- img_feature_proj: 512-dim (after CLIP projection, lives in text-image space)

---

### TF-CLIP

TF-CLIP takes this CLIP backbone and extends it for video-based person re-ID — a significantly
different task. The key challenges it addresses:

1. Multiple frames per tracklet (T=4 frames; need temporal reasoning, not just per-frame)
2. Identity recognition (which of 625 people is this?) — not what CLIP was trained for
3. Camera invariance (same person looks different under different lighting/viewpoints)
4. CLIP's text-image alignment (can be repurposed for re-ID via text prompts like
   "a photo of person 23")

---

### Stage 1: Text-Image alignment (frozen backbone)

Before any re-ID fine-tuning, TF-CLIP first runs a pass through the training set with the backbone
frozen to build a CLIP memory — a set of prototype feature vectors, one per identity class,
computed from the text prompts:

    text prompt: "A photo of a [person_id]"
      → CLIP text encoder → 512-dim text embedding
      → average over all training images of that identity
      → cluster_features[class_id] = 512-dim prototype

This memory serves as a target during training: the image features should match their corresponding
text prototype.

---

### Stage 2: Full fine-tuning (all weights updated)

This is where the actual training happens. Every forward pass through the model produces 4
different feature representations (hence 4 classifier heads):

    Input: [B, T, C, H, W] video batch (B=16 people × 4 frames)

    Step 1: Run each of B×T frames through ViT backbone
      → per-frame: 129 tokens × 768 dim
      → img_feature     = CLS token  [B×T, 768]   — raw ViT features
      → img_feature_proj = projected [B×T, 512]   — in CLIP text-image space

    Step 2: Temporal aggregation (simple mean over T frames)
      img_feature:      [B×T, 768] → reshape [B, T, 768] → mean(T) → [B, 768]
      img_feature_proj: [B×T, 512] → reshape [B, T, 512] → mean(T) → [B, 512]

    Step 3: Temporal Memory Diffusion (TMD) — the main temporal module
      → produces two more representations: cls_f_sp and cls_f_tp (see below)

TMD — Temporal Memory Diffusion is the TF-CLIP paper's key contribution. Rather than just
averaging frames (step 2), it runs a 1-layer transformer-like module across the B×T token sequence
with a special message token mechanism:

    image_features_SAT [B×T, 129, 768]  (all patch tokens, all frames, all videos)
      → CrossFrameAttentionBlock:
          1. For each of the 129 spatial positions, compute a "message token"
             = FC( mean over T frames )  ← a spatial summary across time
          2. Run attention on just the message tokens [T, B, 768] across frames
             ← frames communicate with each other
          3. Append the attended message token back to each frame's patch sequence
          4. Run standard self-attention + MLP on the extended sequence

      → produces cls_f_sp [B×T, 768] and cls_f_tp [B, 768]
         (frame-level temporal features and their temporal mean)

The idea: each frame's patches can "see" a summary of what the other frames in the same tracklet
contain, without fully attending across all B×T×129 tokens (which would be prohibitively expensive).

---

### Camera embedding (SIE)

Person appearance varies dramatically by camera. TF-CLIP handles this with a Side Information
Embedding — a learned 768-dim vector per camera ID that gets added to every patch token before
the transformer:

    cv_embed[camera_id]  →  scaled by sie_coe=1.0
      → repeated for all T frames of that sample
      → added to image input before ViT layers

This biases the visual features based on which camera captured them — effectively teaching the
model "here's a camera-6 image, adjust your representation accordingly."

---

### The 4 classifier heads

After all the feature extraction, four separate nn.Linear heads receive different features and
predict identity class logits:

    classifier2          : img_feature after BN         768 → 625   Main ID loss on raw ViT features
    classifier_proj      : img_feature_proj after BN    512 → 625   ID loss in CLIP projection space
    classifier_proj_temp : cls_f_sp after BN            768 → 625   Frame-level temporal ID loss
    classifier_proj_temp2: cls_f_tp after BN            768 → 625   Temporally aggregated ID loss

The BN (BatchNorm) before each head normalises the features and stabilises training. At test time,
the BN-normalised features (not the raw logits) are what gets used for retrieval.

---

### The losses (3 combined)

    loss = loss1 + loss_frame / T

loss1 combines three things (from the first 3 classifier heads):
- ID loss × 3: CrossEntropy with label smoothing on each head's logits → teaches identity
  discrimination
- Triplet loss: pulls anchor-positive pairs together, pushes anchor-negative pairs apart in
  768-dim feature space → explicitly shapes the retrieval geometry
- I2T loss (Image-to-Text): dot product between video features and all CLIP text prototypes;
  cross-entropy on the identity-matching prototype → keeps the model anchored to CLIP's original
  text-image alignment

loss_frame: CrossEntropy on frame-level head, divided by T to normalise for sequence length →
trains temporal features specifically

The text prototypes (cluster_features) are updated by averaging current image features per class —
so they adapt as training progresses rather than staying fixed at the initial CLIP embeddings.

---

### At test time (retrieval)

    return torch.cat([img_feature, img_feature_proj, cls_f_tp], dim=1)
    # → [B, 768 + 512 + 768] = [B, 2048]

Three features are concatenated and L2-normalised into a single 2048-dim descriptor per tracklet.
Retrieval is then Euclidean distance between query and gallery descriptors — no classifier heads
involved at test time.

The 3 features encode complementary information:
- img_feature (768): discriminative ViT spatial features
- img_feature_proj (512): semantic alignment in CLIP's text space
- cls_f_tp (768): temporally-diffused cross-frame context

---

### Where the quantum modules plug in

Every quantum architecture hooks into this pipeline at the feature level, before the classifier
heads:

- Adapter / interlaced / channel / gated: modify img_feature (768-dim) in-place before it hits
  bottleneck + classifier2
- Frame attention: replaces the .mean(1) temporal aggregation of per-frame features before BN
- QClassifier: replaces the 4 linear heads themselves
- QFeatureExtractor: concatenates VQC output to img_feature before the head
- Quantum kernel: operates entirely at test time on the final 2048-dim descriptor

---
---

## Part 2: Quantum Component Implementations

---

### The shared VQC building block

Every architecture uses the same core pattern inside the quantum part:

    high-dim features (e.g. 768)
      ↓  pre_net: Linear  → compress to n_qubits values
      ↓  sigmoid × π      → squeeze each value into (0, π) as a rotation angle
      ↓  VQC              → angle-encode into qubit states, run StronglyEntanglingLayers,
                             measure qml.probs()
      ↓                     → outputs 2^n probabilities (e.g. 8 qubits → 256 numbers)

Why sigmoid × π?
sigmoid(0) = 0.5 → 0.5π, which is the rotation angle where gradients are maximised for PauliZ.
If we used tanh, the gradient at 0 would be zero — the VQC would be frozen at initialisation.

Why qml.probs()?
It returns the probability of every possible qubit state (all 256 of them for 8 qubits). This
gives the downstream layer a rich feature vector capturing all the entanglement structure, unlike
single-qubit expectations which discard most of it.

---

### Classical ablation (bypass_quantum=True)

Every single architecture has an ablation switch. When bypass_quantum=True:

- The VQC is removed entirely
- It is replaced with nn.Linear(n_qubits → 2^n_qubits) + ReLU — the same input shape and output
  shape
- All the surrounding pre_net, upscale, gate etc. remain identical

This means the ablation tests the exact same architectural structure (bottleneck, residual, etc.)
with a classical linear layer where the circuit was. If VQC results ≈ classical ablation results,
the quantum component is contributing nothing — any gains come from the architecture itself.

---

### 1. QClassifier
Survey taxonomy: Classification level · Sequential (§3.3.1 "Dressed Quantum Circuit")

What it does: The four existing nn.Linear classifier heads (768→625, 512→625 etc.) are each
replaced by a VQC sandwich:

    CLIP features (768)
      → pre_net: Linear(768 → 8)
      → sigmoid × π
      → VQC → 256 probabilities
      → post_net: Linear(256 → 625)

Four separate VQCs run per forward pass. No residual — the output directly becomes the logit.

Why it failed: Without a residual shortcut, all gradient has to flow through the VQC. When VQC
gradients are small (barren plateau), the whole head freezes. The classical ablation
(Linear(8→256)+ReLU) has normal gradients and learns freely, so it always wins.

Survey prediction confirmed: §3.3.1 identifies "classical overshadowing" as the main risk in
sequential architectures. Exactly what happened — classical pre/post-processing layers absorb the
gradient signal, leaving the VQC with nothing to learn.

---

### 2. Q Feature Extractor (parallel)
Survey taxonomy: Feature Extraction level · Parallel (§3.3.2)

What it does: The VQC runs in parallel with the backbone, not instead of it:

    CLIP features (768) ──────────────────────────────────────────→ concatenate → classifier
                         └→ pre_net → sigmoid×π → VQC → 256 probs ↗

The VQC output (256 numbers) is concatenated with the CLIP features (768), giving a 1024-dim
input to the classifier head.

Why it underperformed: The 768-dim classical path dominates the concatenation — the classifier
learns to focus on the classical features and treat the VQC outputs as noise. There's no mechanism
forcing the network to use them.

Survey prediction confirmed: §3.3.2 identifies the "primacy issue" — without careful LR
scheduling, the classical stream dominates. Seen clearly here: qfeatext (0.239) even worse than
the linear probe alone (0.261).

---

### 3. VQC Adapter (plain / residual)
Survey taxonomy: Feature Extraction level · Sequential + residual (§3.3.1)

What it does: The VQC is inserted as a residual correction to the features before the existing
classical classifier heads (which remain unchanged):

    CLIP features (768)  ────────────────────────────────→  +  → adapted features (768)
                         └→ pre_net(768→8) → sigmoid×π →  ↑
                            VQC → 256 probs               upscale: Linear(256→768)

The key property: upscale is initialised near-zero (std=0.001), so at initialisation the adapter
does nothing — it's exactly identity. Training then learns a small correction on top of the
existing features.

Why this works: If the VQC gradient vanishes, the correction stays near zero and the network
degrades gracefully to the unmodified classical baseline. The 4 classifier heads always get clean
gradients via the skip connection.

---

### 4. Dense Angle Adapter
Survey taxonomy: Feature Extraction level · Sequential + residual · Survey Table 3 §3.2.1

Same as the plain adapter, but the encoding step uses both a rotation AND a phase per qubit:

    pre_net: Linear(768 → 16)   ← outputs 2 values per qubit
      first 8  → sigmoid × π    → RY(angle) on each qubit   (rotation around Y axis)
      last 8   → sigmoid × 2π   → RZ(phase) on each qubit   (rotation around Z axis)

This encodes 16 features into 8 qubits instead of 8, effectively doubling the information
capacity. From the survey paper: "Dense Angle Encoding encodes 2N features in N qubits."

Why it helped at 8q but hurt at 4q: At 8 qubits the extra phase gives meaningful additional
signal. At 4 qubits, mapping 8 features through two separate sigmoid scalings (one for angles,
one for phases) is harder to train than just 4 features with one scaling — the pre_net gradients
become noisier.

---

### 5. Channel Attention
Survey taxonomy: Feature Extraction level · Parallel (multiplicative fusion) (§3.3.2)

Instead of an additive residual, the VQC produces per-feature multiplicative attention weights:

    CLIP features (768)
      → pre_net(768→8) → sigmoid×π → VQC → 256 probs
      → expand: Linear(256→768)
      → sigmoid                         → attention weights ∈ (0,1), one per feature
      → output = features × weights + features    (multiplicative × residual +)

The VQC looks at a compressed view of the features and decides which of the 768 channels to
amplify and which to suppress. The + features residual means it can't destroy information, only
gate it.

Init trick: the expand bias is set to +4, so sigmoid(4) ≈ 0.98 ≈ 1 — at initialisation, the
attention weights are near 1 everywhere (no suppression), and training nudges them away from that.

This is why it beat the plain adapter at 15 epochs: the multiplicative path gives the VQC a
different and more natural role — not adding information, but deciding what to amplify.

---

### 6. Frame Attention
Survey taxonomy: Feature Extraction level · Sequential (temporal stage) (§3.3.1)

Instead of attention over feature channels, the VQC produces attention over time (the T=4 frames):

    4 frame features [4 × 768]
      → each frame compressed: pre_net(768→8)
      → each frame through VQC → 256 probs  (4 VQC evals per sample)
      → weight_net: Linear(256→1)    → 1 scalar per frame
      → softmax over 4 frames        → 4 weights summing to 1
      → output = weighted sum of original 4 frame features → 768

This replaces the plain .mean() temporal aggregation. The VQC decides which frames in the
tracklet are most relevant.

At init: weight_net is near-zero, so all 4 weights ≈ equal softmax → same as plain mean.
Training learns to prefer informative frames.

Why it tied the classical ablation: the attention task (pick important frames) may not benefit
from quantum interference specifically — a classical linear layer can learn which frames matter
equally well.

---

### 7. Q-C-Q Interlaced Adapter
Survey taxonomy: Feature Extraction level · Interlaced (§3.3.3)

Two VQC stages with a classical layer between them:

    CLIP features (768)
      Stage 1: pre_net1(768→8) → VQC1 → 256 probs → upscale1(256→256) → ReLU  ← [256]
      Classical: Linear(256→256) → ReLU                                          ← [256]
      Stage 2: pre_net2(256→8) → VQC2 → 256 probs → upscale2(256→768)           ← [768]
      Residual: + original 768 features                                           ← [768]

VQC1 processes the raw CLIP features. The classical middle layer distils them into a 256-dim
latent. VQC2 then processes that latent. The final upscale adds back to the original features
via residual.

The classical layer prevents the two VQCs from stacking gradients — each one only needs to
backpropagate through one VQC stage. This is why it was the strongest architecture: two quantum
processing steps, each with clean gradient paths.

The classical ablation replaces both VQC1 and VQC2 simultaneously with Linear+ReLU; the
classical bottleneck in the middle stays.

Survey prediction: §3.3.3 reports interlaced Q-C-Q as best performer (+5.7% brain tumor). Our
result: tied with classical ablation at 80ep (both 90.7%). Architecture viable but no clear
quantum advantage over a classical two-stage residual.

---

### 8. Quantum Kernel
Survey taxonomy: Postprocessing level · Sequential (§5)

Completely different concept — no VQC in the training loop. After training a classical model
normally, the quantum kernel re-ranks the top-20 gallery results at test time:

    Query feature (768) → pre_net(768→8) → encode as quantum state |ψ_q⟩
    Gallery feature (768) → pre_net(768→8) → encode as quantum state |ψ_g⟩
    Similarity = |⟨ψ_q|ψ_g⟩|²   (IQP fidelity kernel — quantum "dot product")

The pre_net was trained via BCE loss on matching/non-matching pairs. At test time: compute
Euclidean distance for all gallery, take top-20, re-rank those 20 using the quantum similarity
score.

Why it failed: quantum state overlap concentrates near specific values — with 8 qubits the
similarity scores end up in a narrow range [0.0625, 1.0] with almost zero variance. Every pair
looks equally similar to the quantum kernel, so it can't discriminate. The blended version
(λ × quantum + (1−λ) × Euclidean) collapses to pure Euclidean at any λ<1 because the quantum
component has near-zero variance.

    Results: Euclidean 84.1% → random quantum 59.8% → trained quantum 64.3%
             Blended at any λ<1: identical to Euclidean (84.1%)

---

### 9. Gated Adapter
Survey taxonomy: Feature Extraction level · Novel (not in survey taxonomy)

Same as the plain adapter, but adds a learned scalar gate per sample:

    CLIP features (768)
      → gate_net: Linear(768→1) → sigmoid  → g ∈ (0, 1)  [per sample]
      → pre_net(768→8) → VQC → 256 probs → upscale(256→768) → delta
      → output = features + g × delta

If g → 0 everywhere: the VQC correction is suppressed — VQC contributes nothing, and we'd have
a publishable null result showing the model prefers not to use it. If g varies per sample, it
tells us which types of tracklets benefit from quantum processing. This directly addresses the
KIT research question about adaptive routing.

Init: gate_net bias=0 → g=0.5 at init (neutral — half-weight on the quantum correction).

Still pending 80-epoch evaluation.

---
---

## Part 3: Survey Paper Taxonomy Summary

### 2D Taxonomy (from survey paper)

Integration Level: where in the CV pipeline quantum is inserted
  - Preprocessing
  - Feature Extraction   ← where most of our architectures live
  - Classification
  - Postprocessing

Integration Pattern: how classical and quantum interact
  - Sequential: unidirectional flow (classical→quantum or quantum→classical)
  - Parallel: both run concurrently, outputs fused
  - Interlaced: bidirectional alternating between classical and quantum

### All architectures mapped

    Architecture          | Level               | Pattern                  | Survey ref
    ----------------------|---------------------|--------------------------|------------------
    QClassifier           | Classification      | Sequential               | §3.3.1
    Q Feature Extractor   | Feature Extraction  | Parallel (concat)        | §3.3.2
    VQC Adapter           | Feature Extraction  | Sequential + residual    | §3.3.1
    Dense Angle Adapter   | Feature Extraction  | Sequential + residual    | §3.2.1 Table 3
    Channel Attention     | Feature Extraction  | Parallel (multiplicative)| §3.3.2
    Frame Attention       | Feature Extraction  | Sequential (temporal)    | §3.3.1
    Q-C-Q Interlaced      | Feature Extraction  | Interlaced               | §3.3.3
    Quantum Kernel        | Postprocessing      | Sequential               | §5
    Gated Adapter         | Feature Extraction  | Hybrid (novel)           | —

### Survey predictions vs. our results

Sequential (§3.3.1): predicts "classical overshadowing" in sequential architectures
  → CONFIRMED: QClassifier (no residual) always fails. Adapter (residual) resolves it by
    preserving gradient flow through the skip connection.

Parallel (§3.3.2): predicts "primacy issue" — classical stream dominates without LR tuning
  → CONFIRMED: qfeatext classical branch dominates concat. Channel attention (multiplicative)
    avoids it by using a different fusion mechanism rather than concatenation.

Interlaced (§3.3.3): predicts best performance for complex tasks (+5.7% brain tumor in survey)
  → PARTIALLY CONFIRMED: Q-C-Q tied classical ablation at 80ep (both 90.7%). Architecture is
    viable (0.2 pp below TF-CLIP baseline), but no clear quantum advantage over a classical
    two-stage residual with the same structure.

Multi-class degradation (§4.1): predicts performance drops with more classes
  → CONFIRMED: QClassifier fails at 625 classes. Survey notes ~90% binary → ~30% 10-class MNIST.
    Our case is even harder (625 classes). Residual architectures circumvent this by keeping
    classifier heads classical.

Barren plateau (§5): predicts gradient vanishing with more qubits/depth
  → CONFIRMED: 4q 2L (0.309) > 4q 4L (0.299) = 4q 6L (0.299). Classical unaffected.
    Also 12q VQC (0.299) > 8q VQC (0.287) despite having more qubits — because classical 12q
    collapsed (0.250), revealing that at 12q the VQC's inductive bias is the only thing working.
