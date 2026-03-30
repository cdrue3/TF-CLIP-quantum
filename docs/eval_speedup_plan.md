# Deferred: Batch Clip Inference for do_inference_dense

## Status: NOT implemented — saved for future reference

## Problem
`do_inference_dense` processes 1 tracklet at a time (batch_size=1 DataLoader).
- avg ~6 clips/tracklet (MARS, seq_len=4, avg 25 frames)
- GPU forward: ~5-10ms; disk read + PIL decode: ~320ms
- GPU utilization: ~8%. Observed speed: 2.78 tracklet/s

## Why batching is safe
Dense sampling = independent clips, each [seq_len, C, H, W] processed independently
by ViT (temporal attention is within-clip only). Training already batches clips from
different identities in one forward pass (batch_size=8). Concat across tracklets is safe.

## Implementation sketch (NOT done — edit processor_clipreid_stage2.py)

Replace the body of `do_inference_dense` with:

```python
MAX_CLIPS_PER_BATCH = 128  # ~21 tracklets at 6 clips/tracklet avg

buf_clips, buf_cams, buf_views = [], [], []
buf_pids, buf_camids, buf_paths, buf_sizes = [], [], [], []
total_clips_in_buf = 0

def _flush():
    if not buf_clips:
        return
    big = torch.cat(buf_clips, dim=0).to(device)  # [total_clips, s, C, H, W]
    big_cams  = torch.cat(buf_cams,  dim=0).to(device) if buf_cams[0]  is not None else None
    big_views = torch.cat(buf_views, dim=0).to(device) if buf_views[0] is not None else None
    with torch.no_grad():
        feat_all = model(big, cam_label=big_cams, view_label=big_views).cpu()
    offset = 0
    for n_clips, pid, camid, paths in zip(buf_sizes, buf_pids, buf_camids, buf_paths):
        feat_i = feat_all[offset:offset + n_clips]
        evaluator.update((feat_i.mean(0, keepdim=True), pid, camid))
        img_path_list.extend(paths)
        offset += n_clips
    buf_clips.clear(); buf_cams.clear(); buf_views.clear()
    buf_pids.clear(); buf_camids.clear(); buf_paths.clear(); buf_sizes.clear()

pbar_inf = tqdm(val_loader, total=len(val_loader), desc="Inferencing",
                unit="tracklet", dynamic_ncols=True, leave=False)
for img, pid, camid, camids, target_view, imgpath in pbar_inf:
    img = img.cpu()
    if len(img.size()) == 6:
        b, n, s, c, h, w = img.size()
        assert b == 1
        img = img.view(n, s, c, h, w)
    n_clips = img.size(0)

    if cfg.MODEL.SIE_CAMERA and camids is not None:
        camids = camids.cpu()
        if n_clips > camids.size(0):
            camids = torch.repeat_interleave(camids, n_clips // camids.size(0))
    else:
        camids = None

    if cfg.MODEL.SIE_VIEW and target_view is not None:
        target_view = target_view.cpu()
        if n_clips > target_view.size(0):
            target_view = torch.repeat_interleave(target_view, n_clips // target_view.size(0))
    else:
        target_view = None

    buf_clips.append(img); buf_cams.append(camids); buf_views.append(target_view)
    buf_pids.append(pid); buf_camids.append(camid); buf_paths.append(imgpath)
    buf_sizes.append(n_clips)
    total_clips_in_buf += n_clips

    if total_clips_in_buf >= MAX_CLIPS_PER_BATCH:
        _flush()
        total_clips_in_buf = 0

_flush()  # remaining tracklets
```

## Expected outcome
- GPU processes 21+ tracklets of clips per kernel launch vs 1
- GPU utilization: 8% → 40%+
- Speed: ~2.78 t/s → estimated 5-10 t/s (still I/O-limited, but GPU no longer bottleneck)
- Results identical (same mean-feature-per-tracklet)

## File to edit
`processor/processor_clipreid_stage2.py` — `do_inference_dense()` only
No DataLoader, config, or model changes needed.
