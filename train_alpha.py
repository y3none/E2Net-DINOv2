#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案 α — 对标 CamoFormer 训练策略
=====================================
核心原则：
  - 不设独立验证集，全部训练数据用于训练
  - 固定训练 N 个 epoch（默认 100）
  - best model 保存标准：训练 loss 最低的 epoch（而非 val loss）
  - 同时保留每隔 save_freq 个 epoch 的定期 checkpoint

与原版的改动：
  1. 删除 --val_datapath 参数及所有 val 相关代码
  2. best_train_loss 替代 best_val_loss 作为模型选取标准
  3. 训练结束后额外保存 final 权重（最后一个 epoch）
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


from E2Net_dinov2 import E2Net_DINOv2
from dataset import Data, Config


# ═════════════════════════════════════════════════════════════════════════════
# 损失函数
# ═════════════════════════════════════════════════════════════════════════════

def dice_loss(pred, target, smooth=1.0):
    pred   = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    union  = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()

def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target, reduction='mean')

def iou_loss(pred, target, smooth=1.0):
    pred   = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    union  = (pred + target - pred * target).sum(dim=1)
    return 1.0 - ((inter + smooth) / (union + smooth)).mean()

def compute_loss(predictions, masks,
                 lambda_dice=1.0, lambda_bce=1.0, lambda_iou=1.0,
                 lambda_coarse=0.5, lambda_refined=0.3):
    Y_coarse, Y_refined, Y_final = predictions
    masks = masks / 255.0 if masks.max() > 1.0 else masks

    # 主损失（最终预测）
    l_dice_f = dice_loss(Y_final,   masks)
    l_bce_f  = bce_loss(Y_final,    masks)
    l_iou_f  = iou_loss(Y_final,    masks)
    # 辅助损失（粗略）
    l_dice_c = dice_loss(Y_coarse,  masks)
    l_bce_c  = bce_loss(Y_coarse,   masks)
    l_iou_c  = iou_loss(Y_coarse,   masks)
    # 辅助损失（细化）
    l_dice_r = dice_loss(Y_refined, masks)
    l_bce_r  = bce_loss(Y_refined,  masks)
    l_iou_r  = iou_loss(Y_refined,  masks)

    loss_final   = lambda_dice * l_dice_f + lambda_bce * l_bce_f + lambda_iou * l_iou_f
    loss_coarse  = lambda_coarse  * (l_dice_c + l_bce_c + l_iou_c)
    loss_refined = lambda_refined * (l_dice_r + l_bce_r + l_iou_r)
    total        = loss_final + loss_coarse + loss_refined

    loss_dict = {
        'total'      : total.item(),
        'dice_final' : l_dice_f.item(),
        'bce_final'  : l_bce_f.item(),
        'iou_final'  : l_iou_f.item(),
        'loss_coarse': (l_dice_c + l_bce_c + l_iou_c).item(),
        'loss_refined': (l_dice_r + l_bce_r + l_iou_r).item(),
    }
    return total, loss_dict


# ═════════════════════════════════════════════════════════════════════════════
# 训练循环（无 validate）
# ═════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, device, epoch, args):
    model.train()
    epoch_loss = 0.0
    accum = {k: 0.0 for k in
             ['total','dice_final','bce_final','iou_final','loss_coarse','loss_refined']}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for images, masks in pbar:
        images = images.to(device, dtype=torch.float32)
        masks  = masks.to(device,  dtype=torch.float32)

        predictions = model(images)
        # loss, loss_dict = compute_loss(
        #     predictions, masks,
        #     lambda_dice=args.lambda_dice, lambda_bce=args.lambda_bce,
        #     lambda_iou=args.lambda_iou,
        #     lambda_coarse=args.lambda_coarse, lambda_refined=args.lambda_refined,
        # )
        loss, loss_dict = compute_loss(
            predictions, masks,
            lambda_dice=args.lambda_dice, lambda_bce=args.lambda_bce
        )


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        for k in accum:
            if k in loss_dict:
                accum[k] += loss_dict[k]

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{loss_dict['dice_final']:.4f}",
            'bce' : f"{loss_dict['bce_final']:.4f}",
        })

    n = len(dataloader)
    return epoch_loss / n, {k: v / n for k, v in accum.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='方案α — 无验证集训练，对标 CamoFormer'
    )

    # ── 数据（只有训练集，无 val）────────────────────────────────────────
    parser.add_argument('--datapath',         type=str,   default='../dataset/TrainDataset',
                        help='训练集根目录（含 Image/ 和 GT/ 子目录）')

    # ── 训练 ──────────────────────────────────────────────────────────────
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--epochs',           type=int,   default=100,
                        help='固定训练轮数，训练结束即取最终权重')
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--weight_decay',     type=float, default=1e-4)
    parser.add_argument('--image_size',       type=int,   default=392)

    # ── 模型 ──────────────────────────────────────────────────────────────
    parser.add_argument('--encoder_size',     type=str,   default='base',
                        choices=['small','base','large','giant'])
    parser.add_argument('--unified_channels', type=int,   default=256)
    parser.add_argument('--freeze_encoder',   action='store_true', default=True)

    # ── 损失权重 ───────────────────────────────────────────────────────────
    parser.add_argument('--lambda_dice',      type=float, default=1.0)
    parser.add_argument('--lambda_bce',       type=float, default=1.0)
    # parser.add_argument('--lambda_aux',       type=float, default=0.3)
    parser.add_argument('--lambda_iou',       type=float, default=1.0,
                        help='IoU 损失权重（新增，0=关闭）')
    parser.add_argument('--lambda_coarse',    type=float, default=0.5)
    parser.add_argument('--lambda_refined',   type=float, default=0.3)

    # ── Checkpoint ────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint_dir',   type=str,   default='checkpoint/E2Net_alpha')
    parser.add_argument('--resume',           type=str,   default=None)
    parser.add_argument('--save_freq',        type=int,   default=10,
                        help='每隔 N epoch 保存一次定期 checkpoint')
    parser.add_argument('--device',           type=str,   default='cuda')

    args = parser.parse_args()

    # 确保图像尺寸是 14 的倍数（DINOv2 要求）
    if args.image_size % 14 != 0:
        args.image_size = (args.image_size // 14) * 14
        print(f"Image size adjusted to {args.image_size}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("方案 α — 无验证集（对标 CamoFormer）")
    print(f"  训练集    : {args.datapath}")
    print(f"  验证集    : 无（全量数据用于训练）")
    print(f"  保存标准  : 训练 loss 最低的 epoch → best model")
    print(f"  最终权重  : 第 {args.epochs} epoch → final model")
    print("=" * 60)

    # ── 数据（全量训练集，无切分）────────────────────────────────────────
    cfg = Config(datapath=args.datapath, mode='train',
                 snapshot=None, batch_size=args.batch_size)
    train_data   = Data(cfg, 'E2Net')
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == 'cuda'),
        collate_fn=train_data.collate,
    )
    print(f"\nTraining samples: {len(train_data)}  (全量，无验证集切分)")

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = E2Net_DINOv2(
        encoder_size=args.encoder_size,
        freeze_encoder=args.freeze_encoder,
        unified_channels=args.unified_channels,
        # adapter_at=[6, 7, 8, 9, 10, 11]
        adapter_at=[3, 6, 9, 11]
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params — Total:{total:,}  Trainable:{trainable:,}  Frozen:{total-trainable:,}")

    # ── 优化器 / 调度器 ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    # warmup_epochs = 5
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    # cosine_scheduler = CosineAnnealingLR(
    #     optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    # )
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[warmup_epochs]
    # )

    # ── 断点续训 ──────────────────────────────────────────────────────────
    start_epoch      = 0
    best_train_loss  = float('inf')   # ← 方案α 用训练 loss 选 best model

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch     = ckpt['epoch'] + 1
        best_train_loss = ckpt.get('best_train_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_train_loss={best_train_loss:.4f}")

    # ── 训练主循环 ────────────────────────────────────────────────────────
    print(f"\nStart — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, tc = train_epoch(
            model, train_loader, optimizer, device, epoch, args
        )

        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  dice={tc['dice_final']:.4f}  "
              f"bce={tc['bce_final']:.4f}  "
            #   f"iou={tc['iou_final']:.4f}  "
              f"coarse={tc['loss_coarse']:.4f}  "
              f"refined={tc['loss_refined']:.4f}")

        scheduler.step()
        print(f"  LR         : {optimizer.param_groups[0]['lr']:.6f}")

        # ── 定期 checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f'E2Net_alpha_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss'          : train_loss,
                'best_train_loss'     : best_train_loss,
                'args'                : args,
            }, ckpt_path)
            print(f"  Checkpoint : {ckpt_path}")

        # ── best model（训练 loss 最低）───────────────────────────────────
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_path = os.path.join(args.checkpoint_dir, 'E2Net_alpha_best.pth')
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss'          : train_loss,
                'best_train_loss'     : best_train_loss,
                'args'                : args,
            }, best_path)
            print(f"  ✓ Best (train_loss={train_loss:.4f}) → {best_path}")

    # ── 保存最终权重（最后一个 epoch）─────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, 'E2Net_alpha_final.pth')
    torch.save({
        'epoch'               : args.epochs - 1,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss'          : train_loss,
        'best_train_loss'     : best_train_loss,
        'args'                : args,
    }, final_path)
    print(f"\n✓ Final model saved → {final_path}")
    print(f"  Best train loss across all epochs: {best_train_loss:.4f}")
    print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
