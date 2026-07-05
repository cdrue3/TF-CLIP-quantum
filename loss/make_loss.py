# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .quantum_triplet_loss import QuantumTripletLoss
from quantum_models.postprocessing.quantum_label_refiner import QuantumLabelRefiner


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # 1
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore = None, isprint = False):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    # --> 这里需要在采样的时候加上triplet的sampler
                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    # loss = cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        if isinstance(i2tscore, list):
                            I2TLOSS = [xent(scor, target) for scor in i2tscore[0:]]
                            I2TLOSS = sum(I2TLOSS)
                        else:
                            I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                    
                    if isprint:
                        i2t_str = "{:.3f}".format(I2TLOSS) if i2tscore is not None else "skipped"
                        print("Loss: {:.3f}, ID Loss: {:.3f}, TRI Loss: {:.3f}, I2T Loss: {}".format(loss, ID_LOSS, TRI_LOSS, i2t_str))
                        
                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


def make_loss_q_triplet(cfg, num_classes,
                        feat_dim: int = 768,
                        n_qubits: int = 6,
                        n_layers: int = 1):
    """
    Like make_loss but swaps TripletLoss for QuantumTripletLoss.

    Returns:
        loss_func        — same signature as make_loss
        center_criterion — same as make_loss
        q_triplet        — QuantumTripletLoss instance (nn.Module)
                           Needs its own optimizer or attach to model:
                             model.q_triplet = q_triplet   # auto-included in model.parameters()
                           OR:
                             optimizer_qk = torch.optim.Adam(q_triplet.parameters(), lr=1e-3)
                             # call optimizer_qk.step(); optimizer_qk.zero_grad() each batch
    """
    # Build the standard loss (with classical triplet inside)
    loss_func, center_criterion = make_loss(cfg, num_classes)

    # Build the quantum triplet module
    q_triplet = QuantumTripletLoss(
        feat_dim=feat_dim, n_qubits=n_qubits, n_layers=n_layers,
        margin=None if cfg.MODEL.NO_MARGIN else cfg.SOLVER.MARGIN,
    )

    # Wrap loss_func to replace the classical triplet with q_triplet
    _orig_loss_func = loss_func

    def loss_func_q(score, feat, target, target_cam=None, i2tscore=None, isprint=False):
        # Re-implement the triplet portion with the quantum kernel
        # All other terms (ID loss, I2T loss) are computed the same way via _orig_loss_func
        # We call q_triplet directly on the features and swap it in below.

        import torch
        import torch.nn.functional as F_
        from .softmax_loss import CrossEntropyLabelSmooth

        xent_fn = CrossEntropyLabelSmooth(num_classes=num_classes)

        # ID loss
        if isinstance(score, list):
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                ID_LOSS = sum(xent_fn(s, target) for s in score[0:])
            else:
                ID_LOSS = sum(F_.cross_entropy(s, target) for s in score[0:])
        else:
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                ID_LOSS = xent_fn(score, target)
            else:
                ID_LOSS = F_.cross_entropy(score, target)

        # Quantum triplet loss
        if isinstance(feat, list):
            TRI_LOSS = sum(q_triplet(f, target)[0] for f in feat[0:])
        else:
            TRI_LOSS = q_triplet(feat, target)[0]

        loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

        # I2T loss
        if i2tscore is not None:
            if isinstance(i2tscore, list):
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    I2TLOSS = sum(xent_fn(s, target) for s in i2tscore[0:])
                else:
                    I2TLOSS = sum(F_.cross_entropy(s, target) for s in i2tscore[0:])
            else:
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    I2TLOSS = xent_fn(i2tscore, target)
                else:
                    I2TLOSS = F_.cross_entropy(i2tscore, target)
            loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

        if isprint:
            i2t_str = f"{I2TLOSS:.3f}" if i2tscore is not None else "skipped"
            print(f"Loss: {loss:.3f}, ID: {ID_LOSS:.3f}, "
                  f"Q-TRI: {TRI_LOSS:.3f}, I2T: {i2t_str}")

        return loss

    return loss_func_q, center_criterion, q_triplet


def make_loss_qplr(cfg, num_classes,
                   top_k: int = 32,
                   n_qubits: int = 8,
                   n_layers: int = 2,
                   kl_weight: float = 0.5):
    """
    Like make_loss but adds a QuantumLabelRefiner that computes a KL loss
    on the classifier's output logits, capturing inter-class quantum correlations.

    Returns:
        loss_func_qplr  — same signature as make_loss; adds KL term
        center_criterion
        q_refiner       — QuantumLabelRefiner instance (attach to model:
                          model.q_refiner = q_refiner)
    """
    loss_func, center_criterion = make_loss(cfg, num_classes)

    q_refiner = QuantumLabelRefiner(
        num_classes=num_classes,
        top_k=top_k,
        n_qubits=n_qubits,
        n_layers=n_layers,
        kl_weight=kl_weight,
    )

    def loss_func_qplr(score, feat, target, target_cam=None, i2tscore=None, isprint=False):
        # Standard loss
        base_loss = loss_func(score, feat, target, target_cam, i2tscore, isprint)

        # QPLR KL loss on primary classifier head
        primary_score = score[0] if isinstance(score, list) else score
        kl_loss = q_refiner(primary_score, target)

        return base_loss + kl_loss

    return loss_func_qplr, center_criterion, q_refiner
