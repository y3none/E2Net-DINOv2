#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 DINOv2 编码器训练 E2Net 模型
包含完整损失函数与训练流程的脚本
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 导入模型和数据
from E2Net_dinov2_v3 import E2Net_DINOv2
from dataset import Data, Config


def dice_loss(pred, target, smooth=1.0):
    """
    用于分割任务的 Dice 损失函数
    参数:
        pred: [B, 1, H, W] 预测概率图
        target: [B, 1, H, W] 真实掩码（标签）
        smooth: 平滑因子，防止除零
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return 1.0 - dice.mean()


def bce_loss(pred, target):
    """
    二值交叉熵损失
    参数:
        pred: [B, 1, H, W] 预测概率图
        target: [B, 1, H, W] 真实掩码
    """
    bce = F.binary_cross_entropy(pred, target, reduction='mean')
    return bce


def compute_loss(predictions, masks, lambda_dice=1.0, lambda_bce=1.0, 
                lambda_coarse=0.5, lambda_refined=0.3):
    """
    计算 E2Net 的总损失（含多阶段监督）
    参数:
        predictions: (Y_coarse, Y_refined, Y_final) 三个阶段的预测结果
        masks: 真实掩码 [B, 1, H, W]
        lambda_*: 各损失项的权重系数
    返回:
        total_loss: 加权后的总损失
        loss_dict: 各子损失的字典（用于日志记录）
    """
    Y_coarse, Y_refined, Y_final = predictions
    
    # 确保掩码值在 [0, 1] 范围内
    masks = masks / 255.0 if masks.max() > 1.0 else masks
    
    # 主损失：最终预测
    loss_dice_final = dice_loss(Y_final, masks)
    loss_bce_final = bce_loss(Y_final, masks)
    
    # 辅助损失：粗略预测
    loss_dice_coarse = dice_loss(Y_coarse, masks)
    loss_bce_coarse = bce_loss(Y_coarse, masks)
    
    # 辅助损失：细化预测
    loss_dice_refined = dice_loss(Y_refined, masks)
    loss_bce_refined = bce_loss(Y_refined, masks)
    
    # 加权
    loss_final = lambda_dice * loss_dice_final + lambda_bce * loss_bce_final
    loss_coarse = lambda_coarse * (loss_dice_coarse + loss_bce_coarse)
    loss_refined = lambda_refined * (loss_dice_refined + loss_bce_refined)
    
    total_loss = loss_final + loss_coarse + loss_refined
    
    # 构建损失字典
    loss_dict = {
        'total': total_loss.item(),
        'dice_final': loss_dice_final.item(),
        'bce_final': loss_bce_final.item(),
        'dice_coarse': loss_dice_coarse.item(),
        'bce_coarse': loss_bce_coarse.item(),
        'dice_refined': loss_dice_refined.item(),
        'bce_refined': loss_bce_refined.item()
    }
    
    return total_loss, loss_dict


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """
    执行一个训练周期
    """
    model.train()
    
    epoch_loss = 0
    loss_components = {
        'total': 0, 'dice_final': 0, 'bce_final': 0,
        'dice_coarse': 0, 'bce_coarse': 0,
        'dice_refined': 0, 'bce_refined': 0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        # 前向传播
        predictions = model(images)
        
        # 计算损失
        loss, loss_dict = compute_loss(
            predictions, masks,
            lambda_dice=args.lambda_dice,
            lambda_bce=args.lambda_bce,
            lambda_coarse=args.lambda_coarse,
            lambda_refined=args.lambda_refined
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累计损失
        epoch_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{loss_dict['dice_final']:.4f}",
            'bce': f"{loss_dict['bce_final']:.4f}"
        })
    
    # 计算平均损失
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return epoch_loss, loss_components


def validate(model, dataloader, device, args):
    """
    模型验证
    """
    model.eval()
    
    total_loss = 0
    loss_components = {
        'total': 0, 'dice_final': 0, 'bce_final': 0
    }
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, masks in pbar:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            loss, loss_dict = compute_loss(
                predictions, masks,
                lambda_dice=args.lambda_dice,
                lambda_bce=args.lambda_bce,
                lambda_coarse=args.lambda_coarse,
                lambda_refined=args.lambda_refined
            )
            
            total_loss += loss.item()
            for key in ['dice_final', 'bce_final']:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for key in loss_components:
        loss_components[key] /= num_batches if num_batches > 0 else 1
    
    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser(description='Train E2Net with DINOv2')
    
    # 数据参数
    parser.add_argument('--datapath', type=str, default='dataset/TrainDataset',
                        help='Path to training dataset')
    parser.add_argument('--val_datapath', type=str, default='dataset/TestDataset/CAMO',
                        help='Path to validation dataset')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay')
    
    # 模型参数
    parser.add_argument('--encoder_size', type=str, default='base',
                        choices=['small', 'base', 'large', 'giant'],
                        help='DINOv2 model size')
    parser.add_argument('--unified_channels', type=int, default=256, 
                        help='Unified channel dimension')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='Freeze DINOv2 encoder')
    
    # 损失权重
    parser.add_argument('--lambda_dice', type=float, default=1.0, 
                        help='Dice loss weight')
    parser.add_argument('--lambda_bce', type=float, default=1.0, 
                        help='BCE loss weight')
    parser.add_argument('--lambda_coarse', type=float, default=0.5, 
                        help='Coarse loss weight')
    parser.add_argument('--lambda_refined', type=float, default=0.3, 
                        help='Refined loss weight')
    
    # Checkpoint 参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/E2Net_DINOv2',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=10, 
                        help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # 图像尺寸
    parser.add_argument('--image_size', type=int, default=392,
                        help='Image size (must be multiple of 14 for DINOv2)')
    
    args = parser.parse_args()
    
    # 验证图像尺寸
    if args.image_size % 14 != 0:
        print(f"Warning: Image size {args.image_size} is not multiple of 14!")
        args.image_size = (args.image_size // 14) * 14
        print(f"Adjusted to {args.image_size}")
    
    # 创建 checkpoint 文件夹
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据集
    print("\nLoading datasets...")
    
    # 训练数据集
    cfg_train = Config(
        datapath=args.datapath,
        mode='train',
        snapshot=None,
        batch_size=args.batch_size
    )
    train_data = Data(cfg_train, 'E2Net')
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=train_data.collate
    )
    
    # 验证数据集
    cfg_val = Config(
        datapath=args.val_datapath,
        mode='train',
        snapshot=None,
        batch_size=1
    )
    val_data = Data(cfg_val, 'E2Net')
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=val_data.collate
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # 创建模型
    print("\nCreating E2Net with DINOv2...")
    model = E2Net_DINOv2(
        encoder_size=args.encoder_size,
        freeze_encoder=args.freeze_encoder,
        unified_channels=args.unified_channels
    )
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    # 优化器（仅优化可训练参数）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 从检查点恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # 开始训练
    print("\n" + "="*70)
    print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_loss, train_loss_components = train_epoch(
            model, train_loader, optimizer, device, epoch, args
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print("  Components:")
        for key, value in train_loss_components.items():
            if key != 'total':
                print(f"    {key}: {value:.4f}")
        
        # 验证
        val_loss, val_loss_components = validate(model, val_loader, device, args)
        print(f"\nValidation Loss: {val_loss:.4f}")
        print("  Components:")
        for key, value in val_loss_components.items():
            if key != 'total':
                print(f"    {key}: {value:.4f}")
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nLearning rate: {current_lr:.6f}")
        
        # 保存 checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'E2Net_DINOv2_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': args
            }, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # 保存 best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'E2Net_DINOv2_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': args
            }, best_model_path)
            print(f"✓ Best model updated: {best_model_path} (val_loss: {val_loss:.4f})")
    
    print("\n" + "="*70)
    print(f"Training Completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()