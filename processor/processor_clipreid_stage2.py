import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.iotools import save_checkpoint, save_slim_checkpoint
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from tqdm import tqdm

# ==================================================================================
# Training Routine (Stage 2)
# ==================================================================================
def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage1,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, num_classes):
    
    # --- Configuration & Device Setup ---
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    # --- Logger Initialization ---
    logger = logging.getLogger("TFCLIP.train")
    logger.info('start training')
    
    # --- Distributed Data Parallel (DDP) Setup ---
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_id1 = AverageMeter()
    acc_meter_id2 = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent_frame = CrossEntropyLabelSmooth(num_classes=num_classes)

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        import collections
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers

    # --- Stage 1: CLIP-Memory Initialization (with disk cache) ---
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # Cache is keyed by dataset root so different datasets don't collide.
    _cache_path = os.path.join(cfg.DATASETS.ROOT_DIR, 'clip_memory_cache.pt')
    if os.path.exists(_cache_path):
        print(f"=> Loading cached CLIP-Memory from {_cache_path}")
        cluster_features = torch.load(_cache_path, map_location='cuda').detach()
    else:
        print("=> Automatically generating CLIP-Memory (might take a while, have a coffee)")
        image_features = []
        labels = []

        with torch.no_grad():
            for n_iter, (img, vid, target_cam, target_view) in tqdm(
                enumerate(train_loader_stage1),
                total=len(train_loader_stage1),
                desc="CLIP-Memory",
                unit="batch",
                dynamic_ncols=True,
            ):
                img = img.to(device)
                target = vid.to(device)

                if len(img.size()) == 6:
                    b, n, s, c, h, w = img.size()
                    assert (b == 1)
                    img = img.view(b * n, s, c, h, w)

                    # Chunk to avoid OOM on dense-sampled tracklets (e.g. 937 frames).
                    MAX_CLIP_BATCH = 16
                    chunk_feats = []
                    with amp.autocast(enabled=True):
                        for ci in range(0, n, MAX_CLIP_BATCH):
                            img_chunk = img[ci : ci + MAX_CLIP_BATCH].to(device)
                            chunk_feats.append(model(img_chunk, get_image=True).cpu())
                        image_feature = torch.cat(chunk_feats, dim=0)
                        image_feature = image_feature.view(-1, image_feature.size(1))
                        image_feature = torch.mean(image_feature, 0, keepdim=True)

                        for i, img_feat in zip(target, image_feature):
                            labels.append(i)
                            image_features.append(img_feat.cpu())
                else:
                    with amp.autocast(enabled=True):
                        image_feature = model(img, get_image=True)
                        for i, img_feat in zip(target, image_feature):
                            labels.append(i)
                            image_features.append(img_feat.cpu())

            labels_list = torch.stack(labels, dim=0).cuda()
            image_features_list = torch.stack(image_features, dim=0).cuda()

        cluster_features = generate_cluster_features(labels_list.cpu().numpy(), image_features_list).detach()
        torch.save(cluster_features.cpu(), _cache_path)
        print(f"=> Saved CLIP-Memory cache to {_cache_path}")

    best_performance = 0.0
    best_epoch = 1

    # ==============================================================================
    # Main Training Loop
    # ==============================================================================
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        acc_meter_id1.reset()
        acc_meter_id2.reset()
        evaluator.reset()

        model.train()

        pbar = tqdm(
            enumerate(train_loader_stage2),
            total=len(train_loader_stage2),
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for n_iter, (img, vid, target_cam, target_view) in pbar:
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = img.to(device)
            target = vid.to(device)
            
            target_cam = target_cam.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            
            with amp.autocast(enabled=True):
                B, T, C, H, W = img.shape

                score, feat, logits1 = model(x = img, cam_label=target_cam, view_label=target_view, text_features2=cluster_features)

                score1 = score[0:3]
                score2 = score[3]

                # Guard: if cluster_features covers fewer classes than the full
                # training set (e.g. when --max_mem_batches limits stage1),
                # skip the I2T loss to avoid out-of-bounds indexing.
                i2t_arg = logits1 if logits1.size(1) >= score[0].size(1) else None

                if (n_iter + 1) % log_period == 0:
                    loss1 = loss_fn(score1, feat, target, target_cam, i2t_arg, isprint=True)
                else:
                    loss1 = loss_fn(score1, feat, target, target_cam, i2t_arg)

                targetX = target.unsqueeze(1).expand(B, T).contiguous().view(B * T, -1).squeeze(1)
                loss_frame = xent_frame(score2, targetX)
                
                loss = loss1 + loss_frame / T

            scaler.scale(loss).backward()

            # ── One-time gradient diagnostic (first iteration only) ──────────
            if n_iter == 0 and epoch == 1:
                logger.info("[GradCheck] Gradient norms after first backward pass:")
                for name, param in model.named_parameters():
                    if 'classifier' in name:
                        if param.grad is not None:
                            logger.info(f"  {name}: grad_norm={param.grad.norm().item():.4e}  param_norm={param.data.norm().item():.4e}")
                        else:
                            logger.info(f"  {name}: grad=None  (no gradient!)")
            # ─────────────────────────────────────────────────────────────────

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc1 = (logits1.max(1)[1] == target).float().mean()
            acc_id1 = (score[0].max(1)[1] == target).float().mean()
            acc_id2 = (score[3].max(1)[1] == targetX).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc1, 1)
            acc_meter_id1.update(acc_id1, 1)
            acc_meter_id2.update(acc_id2, 1)

            # Update tqdm bar with running averages every iteration
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.3f}",
                acc_clip=f"{acc_meter.avg:.3f}",
                acc_id=f"{acc_meter_id1.avg:.3f}",
                lr=f"{scheduler.get_lr()[0]:.1e}",
            )

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc_clip: {:.3f}, Acc_id1: {:.3f}, Acc_id2: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader_stage2),
                            loss_meter.avg, acc_meter.avg, acc_meter_id1.avg, acc_meter_id2.avg, scheduler.get_lr()[0]))

        scheduler.step() 

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN and dist.get_rank() != 0:
                pass 
            else:
                model.eval()
                # Redirects to the dense inference function 
                do_inference_dense(cfg, model, val_loader, num_query)
            
            save_slim_checkpoint(model, fpath=os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


# ==================================================================================
# Inference Function (Dense Mode) - FORCED CPU SAFETY
# ==================================================================================
def do_inference_dense(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    # 16 is a safe, efficient size for models that retain T-dim features (e.g. frame attention).
    MAX_INFERENCE_BATCH = 16

    pbar_inf = tqdm(
        val_loader,
        total=len(val_loader),
        desc="Inferencing",
        unit="tracklet",
        dynamic_ncols=True,
        leave=False,
    )
    for img, pid, camid, camids, target_view, imgpath in pbar_inf:
        # --- NUCLEAR OPTION: FORCE CPU ---
        # If the loader put this on GPU, move it back to CPU immediately.
        img = img.cpu()
        
        # 1. Reshape to 5D [Batch * Clips, Seq_Len, C, H, W] on CPU
        if len(img.size()) == 6:
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)


        # 2. Dynamic Label Alignment (On CPU)
        n_imgs = img.size(0)

        # Fix Camera Labels 
        if cfg.MODEL.SIE_CAMERA and camids is not None:
            camids = camids.cpu() # Ensure CPU
            n_labels = camids.size(0)
            if n_imgs > n_labels:
                repeats = n_imgs // n_labels
                camids = torch.repeat_interleave(camids, repeats)
        else:
            camids = None

        # Fix View Labels 
        if cfg.MODEL.SIE_VIEW and target_view is not None:
            target_view = target_view.cpu() # Ensure CPU
            n_labels = target_view.size(0)
            if n_imgs > n_labels:
                repeats = n_imgs // n_labels
                target_view = torch.repeat_interleave(target_view, repeats)
        else:
            target_view = None

        # 3. Chunked Forward Pass
        feat_list = []
        
        with torch.no_grad():
            for i in range(0, n_imgs, MAX_INFERENCE_BATCH):
                # A. Slice the tensor on CPU (Fast, Low Memory)
                img_chunk = img[i : i + MAX_INFERENCE_BATCH]
                
                # B. Move ONLY this small chunk to GPU
                img_chunk = img_chunk.to(device)
                
                # Handle labels
                cam_chunk = None
                if camids is not None:
                    cam_chunk = camids[i : i + MAX_INFERENCE_BATCH].to(device)
                
                view_chunk = None
                if target_view is not None:
                    view_chunk = target_view[i : i + MAX_INFERENCE_BATCH].to(device)
                
                # C. Run Model
                feat_out = model(img_chunk, cam_label=cam_chunk, view_label=view_chunk)
                
                # D. Immediately move results back to CPU
                feat_list.append(feat_out.cpu())
            
            # 4. Aggregate Results (on CPU)
            feat = torch.cat(feat_list, dim=0)
            feat = feat.view(-1, feat.size(1))
            feat = torch.mean(feat, 0, keepdim=True)
            
            evaluator.update((feat, pid, camid))
            if imgpath is not None:
                img_path_list.extend(imgpath)

    logger.info("Computing metrics...")
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


# ==================================================================================
# Inference Function (RRS Mode)
# ==================================================================================
def do_inference_rrs(cfg,
                     model,
                     val_loader,
                     num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)
        
        if len(img.size()) == 6:
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)

        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            
            feat = model(img, cam_label=camids, view_label=target_view)
            
            evaluator.update((feat, pid, camid))
            if imgpath is not None:
                img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]