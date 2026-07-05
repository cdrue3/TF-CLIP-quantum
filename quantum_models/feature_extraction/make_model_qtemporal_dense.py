"""
quantum_models/make_model_qtemporal_dense.py

TF-CLIP model variant — Temporal Quantum Aggregation (Dense-TQA).

What changed vs. make_model_adapter.py
---------------------------------------
- QuantumTemporalDense replaces the mean-pool temporal aggregation in the primary path:
      img_feature.view(B, T, 768).mean(1)   →   self.tqa(img_feature.view(B, T, 768))
  Dense-TQA encodes T frames sequentially into a shared VQC (data re-uploading over time),
  then applies a skip connection: output = mean_pool(x) + upscale(VQC(...)).

- All 4 classifier heads remain standard nn.Linear (no VQC in classification).

- img_feature_proj (512-dim) and cls_f_tp (TMD output) retain plain mean(1).

- The later quantum_adapter application block from make_model_adapter.py is removed;
  Dense-TQA is already applied at the pooling step, so feat = self.bottleneck(img_feature).

Why this differs from the adapter approach:
  adapter: img_feature (post-mean-pool, 768) → VQC_residual → BN → nn.Linear
      VQC corrects an already-pooled feature; temporal ordering already lost.
  Dense-TQA (this file): [B, T, 768] → VQC sequential upload → [B, 768] → BN → nn.Linear
      VQC sees all T frames; temporal ordering and interference are preserved.
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from collections import OrderedDict

from model.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.clip.model import QuickGELU, LayerNorm
from model.clip import clip
from model.Visual_Prompt import visual_prompt

# Temporal quantum aggregation (new)
from quantum_models.feature_extraction.quantum_temporal_dense import QuantumTemporalDense

_tokenizer = _Tokenizer()


# ============================================================================
# Utility weight initialisers  (identical to original)
# ============================================================================

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# ============================================================================
# CLIP loader  (identical to original)
# ============================================================================

def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(
        state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size
    )
    return model


# ============================================================================
# Cross-frame attention block  (identical to original)
# ============================================================================

class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 droppath=0., T=0):
        super().__init__()
        self.T = T

        self.message_fc   = nn.Linear(d_model, d_model)
        self.message_ln   = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head)

        self.attn     = nn.MultiheadAttention(d_model, n_head)
        self.ln_1     = LayerNorm(d_model)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp      = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model * 4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2      = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d)

        msg_token = self.message_fc(x.mean(0))
        msg_token = msg_token.view(b, self.T, 1, d)
        msg_token = msg_token.permute(1, 2, 0, 3).view(self.T, b, d)
        msg_token = msg_token + self.drop_path(
            self.message_attn(
                self.message_ln(msg_token),
                self.message_ln(msg_token),
                self.message_ln(msg_token),
                need_weights=False,
            )[0]
        )
        msg_token = msg_token.view(self.T, 1, b, d).permute(1, 2, 0, 3)

        x = torch.cat([x, msg_token], dim=0)
        x = x.view(l + 1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


# ============================================================================
# Temporal Memory Diffusion  (identical to original)
# ============================================================================

class Temporal_Memory_Difusion(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: torch.Tensor = None, droppath=None, T=8):
        super().__init__()
        if droppath is None:
            droppath = [0.0] * layers
        self.width  = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T)
              for i in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# ============================================================================
# Multi-head attention + prompt generator  (identical to original)
# ============================================================================

class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5

        self.q_proj    = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj    = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj    = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn  = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.mlp        = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ImageSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1):
        super().__init__()
        self.norm         = nn.LayerNorm(embed_dim)
        self.memory_proj  = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.text_proj    = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj     = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.decoder      = nn.ModuleList(
            [PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)]
        )
        self.alpha        = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        visual = self.memory_proj(visual)
        text   = self.text_proj(text)
        for layer in self.decoder:
            text = layer(text, visual)
        text = self.out_proj(text)
        return text


# ============================================================================
# Main model  — TEMPORAL QUANTUM AGGREGATION VARIANT
# ============================================================================

class build_transformer(nn.Module):
    """
    TF-CLIP build_transformer with Temporal Quantum Aggregation on the primary path.

    Differences from make_model_adapter.py:
        - tqa: QuantumTemporalDense — replaces img_feature.mean(1) in the primary path.
          Input [B, T, 768] → VQC sequential upload → skip → [B, 768].
          bypass_quantum=True: plain mean(1) (classical ablation).
        - No separate quantum_adapter step after pooling (Dense-TQA is the pooling step).
        - classifier2, classifier_proj, classifier_proj_temp, classifier_proj_temp2:
          all remain classical nn.Linear (unchanged).
        - img_feature_proj (512-dim) and cls_f_tp (TMD output) retain plain mean(1).

    Forward path (primary, classifier2):
        [B, T, 768] → tqa → [B, 768] → bottleneck (BN) → classifier2 (nn.Linear)

    Forward path (other 3 heads): unchanged from original TF-CLIP.
    """

    def __init__(self, num_classes, camera_num, view_num, cfg,
                 n_qubits: int = 8, n_layers: int = 2, bypass_quantum: bool = False):
        super().__init__()
        self.model_name  = cfg.MODEL.NAME
        self.cos_layer   = cfg.MODEL.COS_LAYER
        self.neck        = cfg.MODEL.NECK
        self.neck_feat   = cfg.TEST.NECK_FEAT

        if self.model_name == 'ViT-B-16':
            self.in_planes      = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes      = 2048
            self.in_planes_proj = 1024

        self.num_classes = num_classes
        self.camera_num  = camera_num
        self.view_num    = view_num
        self.sie_coe     = cfg.MODEL.SIE_COE

        # ------------------------------------------------------------------ #
        # Temporal Quantum Aggregation  (NEW)
        # Replaces img_feature.view(B, T, 768).mean(1) in the primary path.
        # Skip: output = mean_pool(x) + upscale(VQC(pre_net(x)))
        # bypass_quantum=True: output = x.mean(1) (plain mean-pool ablation)
        # ------------------------------------------------------------------ #
        self.tqa = QuantumTemporalDense(
            in_features=self.in_planes,
            n_qubits=n_qubits,
            n_layers=n_layers,
            seq_len=cfg.INPUT.SEQ_LEN,
            bypass_quantum=bypass_quantum,
        )

        # ------------------------------------------------------------------ #
        # Classifier heads — all classical nn.Linear  (unchanged from original)
        # ------------------------------------------------------------------ #
        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.classifier_proj_temp = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp.apply(weights_init_classifier)

        self.classifier_proj_temp2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp2.apply(weights_init_classifier)

        # ------------------------------------------------------------------ #
        # Bottleneck layers  (identical to original)
        # ------------------------------------------------------------------ #
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.bottleneck_proj_temp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp.bias.requires_grad_(False)
        self.bottleneck_proj_temp.apply(weights_init_kaiming)

        self.bottleneck_proj_temp2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp2.bias.requires_grad_(False)
        self.bottleneck_proj_temp2.apply(weights_init_kaiming)

        # ------------------------------------------------------------------ #
        # Image encoder  (identical to original)
        # ------------------------------------------------------------------ #
        self.h_resolution      = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution      = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(
            self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size
        )
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        # ------------------------------------------------------------------ #
        # SIE camera/view embeddings  (identical to original)
        # ------------------------------------------------------------------ #
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        # ------------------------------------------------------------------ #
        # Temporal modules  (identical to original)
        # ------------------------------------------------------------------ #
        self.SSP = ImageSpecificPrompt()
        self.TMD = Temporal_Memory_Difusion(
            width=768, layers=1, heads=12, droppath=None, T=cfg.INPUT.SEQ_LEN
        )

        print(
            f"[Dense-TQA] n_qubits={n_qubits}, n_layers={n_layers}, seq_len={cfg.INPUT.SEQ_LEN}  "
            f"(replaces mean-pool on primary path; all 4 heads: classical nn.Linear)"
        )

    # ---------------------------------------------------------------------- #
    def forward(self, x=None, get_image=False, cam_label=None, view_label=None,
                text_features2=None):

        B, T, C, H, W = x.shape

        if get_image:
            x = x.view(-1, C, H, W)
            image_features, image_features_proj = self.image_encoder(x)

            if self.model_name == 'RN50':
                img_feature_proj = image_features_proj[0].view(B, T, -1).mean(1)
                return img_feature_proj
            elif self.model_name == 'ViT-B-16':
                img_feature_proj = image_features_proj[:, 0].view(B, T, -1).mean(1)
                return img_feature_proj

        if self.model_name == 'RN50':
            x = x.view(-1, C, H, W)
            image_features_last, image_features, image_features_proj = self.image_encoder(x)

            img_feature_last = nn.functional.avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(
                image_features, image_features.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

            img_feature_last = img_feature_last.view(B, T, -1).mean(1)
            img_feature      = img_feature.view(B, T, -1).mean(1)
            img_feature_proj = img_feature_proj.view(B, T, -1).mean(1)

        elif self.model_name == 'ViT-B-16':
            x = x.view(-1, C, H, W)

            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)
            image_features, image_features_proj_raw = self.image_encoder(x, cv_embed)

            img_feature      = image_features[:, 0]
            img_feature_proj = image_features_proj_raw[:, 0]

            # ── PRIMARY TEMPORAL AGGREGATION ─────────────────────────────── #
            # Dense-TQA replaces .mean(1); bypass_quantum=True gives plain mean-pool.
            img_feature      = self.tqa(img_feature.view(B, T, -1))       # [B, 768]
            img_feature_proj = img_feature_proj.view(B, T, -1).mean(1)   # [B, 512] unchanged
            # ─────────────────────────────────────────────────────────────── #

            ft_for_another_branch = image_features.detach()
            image_features_SAT    = ft_for_another_branch.permute(1, 0, 2)

            f_sp  = self.TMD(image_features_SAT)
            f_sp2 = f_sp.permute(1, 0, 2)

            cls_f_sp     = f_sp2.mean(1)
            cls_f_sp_tap = cls_f_sp.view(B, T, -1)
            cls_f_tp     = cls_f_sp_tap.mean(1)

        # ------------------------------------------------------------------ #
        # Bottleneck projections.
        # Dense-TQA is already applied above; just pass img_feature through BN.
        # ------------------------------------------------------------------ #
        feat            = self.bottleneck(img_feature)
        feat_proj       = self.bottleneck_proj(img_feature_proj)
        feat_proj_frame = self.bottleneck_proj_temp(cls_f_sp)
        feat_proj_temp  = self.bottleneck_proj_temp2(cls_f_tp)

        if self.training:
            text_features2 = text_features2.unsqueeze(0).expand(B, -1, -1)
            image_features_proj_raw2 = image_features_proj_raw.view(
                B, T, -1, image_features_proj_raw.shape[-1]
            )
            video_feature_project = image_features_proj_raw2.mean(1)
            text_features2 = text_features2 + self.SSP(text_features2, video_feature_project)
            logits = torch.einsum("bd,bkd->bk", img_feature_proj, text_features2)

            cls_score            = self.classifier2(feat)
            cls_score_proj       = self.classifier_proj(feat_proj)
            cls_score_proj_frame = self.classifier_proj_temp(feat_proj_frame)
            cls_score_proj_temp  = self.classifier_proj_temp2(feat_proj_temp)

            return (
                [cls_score, cls_score_proj, cls_score_proj_temp, cls_score_proj_frame],
                [img_feature, img_feature_proj, cls_f_tp],
                logits,
            )

        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj, cls_f_tp], dim=1)

    # ---------------------------------------------------------------------- #
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


# ============================================================================
# Factory function
# ============================================================================

def make_model(cfg, num_class, camera_num, view_num,
               n_qubits: int = 8, n_layers: int = 2,
               bypass_quantum: bool = False):
    """
    Build the TF-CLIP model with Temporal Quantum Aggregation on the primary path.

    Args:
        cfg            : YACS config node.
        num_class      : Number of identity classes.
        camera_num     : Number of cameras.
        view_num       : Number of views.
        n_qubits       : Qubit count for Dense-TQA. Default 8.
        n_layers       : VQC depth. Default 2.
        bypass_quantum : If True, Dense-TQA returns plain mean-pool (classical ablation).

    Returns:
        build_transformer instance with QuantumTemporalDense + classical classifier heads.
    """
    model = build_transformer(
        num_class, camera_num, view_num, cfg,
        n_qubits=n_qubits, n_layers=n_layers,
        bypass_quantum=bypass_quantum,
    )
    return model
