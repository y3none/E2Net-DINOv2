# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # """
# # 使用 DINOv2 编码器训练 E2Net 模型
# # 包含完整损失函数与训练流程的脚本
# # """

# # import os
# # import sys
# # import argparse
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader
# # import numpy as np
# # from tqdm import tqdm
# # from datetime import datetime

# # # 导入模型和数据
# # from E2Net_dinov2 import E2Net_DINOv2
# # from dataset import Data, Config


# # def dice_loss(pred, target, smooth=1.0):
# #     """
# #     用于分割任务的 Dice 损失函数
# #     参数:
# #         pred: [B, 1, H, W] 预测概率图
# #         target: [B, 1, H, W] 真实掩码（标签）
# #         smooth: 平滑因子，防止除零
# #     """
# #     pred = pred.contiguous().view(pred.size(0), -1)
# #     target = target.contiguous().view(target.size(0), -1)
    
# #     intersection = (pred * target).sum(dim=1)
# #     union = pred.sum(dim=1) + target.sum(dim=1)
    
# #     dice = (2.0 * intersection + smooth) / (union + smooth)
    
# #     return 1.0 - dice.mean()


# # def bce_loss(pred, target):
# #     """
# #     二值交叉熵损失
# #     参数:
# #         pred: [B, 1, H, W] 预测概率图
# #         target: [B, 1, H, W] 真实掩码
# #     """
# #     bce = F.binary_cross_entropy(pred, target, reduction='mean')
# #     return bce


# # def compute_loss(predictions, masks, lambda_dice=1.0, lambda_bce=1.0, 
# #                 lambda_coarse=0.5, lambda_refined=0.3):
# #     """
# #     计算 E2Net 的总损失（含多阶段监督）
# #     参数:
# #         predictions: (Y_coarse, Y_refined, Y_final) 三个阶段的预测结果
# #         masks: 真实掩码 [B, 1, H, W]
# #         lambda_*: 各损失项的权重系数
# #     返回:
# #         total_loss: 加权后的总损失
# #         loss_dict: 各子损失的字典（用于日志记录）
# #     """
# #     Y_coarse, Y_refined, Y_final = predictions
    
# #     # 确保掩码值在 [0, 1] 范围内
# #     masks = masks / 255.0 if masks.max() > 1.0 else masks
    
# #     # 主损失：最终预测
# #     loss_dice_final = dice_loss(Y_final, masks)
# #     loss_bce_final = bce_loss(Y_final, masks)
    
# #     # 辅助损失：粗略预测
# #     loss_dice_coarse = dice_loss(Y_coarse, masks)
# #     loss_bce_coarse = bce_loss(Y_coarse, masks)
    
# #     # 辅助损失：细化预测
# #     loss_dice_refined = dice_loss(Y_refined, masks)
# #     loss_bce_refined = bce_loss(Y_refined, masks)
    
# #     # 加权
# #     loss_final = lambda_dice * loss_dice_final + lambda_bce * loss_bce_final
# #     loss_coarse = lambda_coarse * (loss_dice_coarse + loss_bce_coarse)
# #     loss_refined = lambda_refined * (loss_dice_refined + loss_bce_refined)
    
# #     total_loss = loss_final + loss_coarse + loss_refined
    
# #     # 构建损失字典
# #     loss_dict = {
# #         'total': total_loss.item(),
# #         'dice_final': loss_dice_final.item(),
# #         'bce_final': loss_bce_final.item(),
# #         'dice_coarse': loss_dice_coarse.item(),
# #         'bce_coarse': loss_bce_coarse.item(),
# #         'dice_refined': loss_dice_refined.item(),
# #         'bce_refined': loss_bce_refined.item()
# #     }
    
# #     return total_loss, loss_dict


# # def train_epoch(model, dataloader, optimizer, device, epoch, args):
# #     """
# #     执行一个训练周期
# #     """
# #     model.train()
    
# #     epoch_loss = 0
# #     loss_components = {
# #         'total': 0, 'dice_final': 0, 'bce_final': 0,
# #         'dice_coarse': 0, 'bce_coarse': 0,
# #         'dice_refined': 0, 'bce_refined': 0
# #     }
    
# #     pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
# #     for batch_idx, (images, masks) in enumerate(pbar):
# #         images = images.to(device, dtype=torch.float32)
# #         masks = masks.to(device, dtype=torch.float32)
        
# #         # 前向传播
# #         predictions = model(images)
        
# #         # 计算损失
# #         loss, loss_dict = compute_loss(
# #             predictions, masks,
# #             lambda_dice=args.lambda_dice,
# #             lambda_bce=args.lambda_bce,
# #             lambda_coarse=args.lambda_coarse,
# #             lambda_refined=args.lambda_refined
# #         )
        
# #         # 反向传播
# #         optimizer.zero_grad()
# #         loss.backward()
        
# #         # 梯度裁剪（防止梯度爆炸）
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
# #         optimizer.step()
        
# #         # 累计损失
# #         epoch_loss += loss.item()
# #         for key in loss_components:
# #             if key in loss_dict:
# #                 loss_components[key] += loss_dict[key]
        
# #         # 更新进度条
# #         pbar.set_postfix({
# #             'loss': f"{loss.item():.4f}",
# #             'dice': f"{loss_dict['dice_final']:.4f}",
# #             'bce': f"{loss_dict['bce_final']:.4f}"
# #         })
    
# #     # 计算平均损失
# #     num_batches = len(dataloader)
# #     epoch_loss /= num_batches
# #     for key in loss_components:
# #         loss_components[key] /= num_batches
    
# #     return epoch_loss, loss_components


# # def validate(model, dataloader, device, args):
# #     """
# #     模型验证
# #     """
# #     model.eval()
    
# #     total_loss = 0
# #     loss_components = {
# #         'total': 0, 'dice_final': 0, 'bce_final': 0
# #     }
    
# #     with torch.no_grad():
# #         pbar = tqdm(dataloader, desc='Validation')
        
# #         for images, masks in pbar:
# #             images = images.to(device, dtype=torch.float32)
# #             masks = masks.to(device, dtype=torch.float32)
            
# #             # 前向传播
# #             predictions = model(images)
            
# #             # 计算损失
# #             loss, loss_dict = compute_loss(
# #                 predictions, masks,
# #                 lambda_dice=args.lambda_dice,
# #                 lambda_bce=args.lambda_bce,
# #                 lambda_coarse=args.lambda_coarse,
# #                 lambda_refined=args.lambda_refined
# #             )
            
# #             total_loss += loss.item()
# #             for key in ['dice_final', 'bce_final']:
# #                 if key in loss_dict:
# #                     loss_components[key] += loss_dict[key]
            
# #             pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    
# #     num_batches = len(dataloader)
# #     avg_loss = total_loss / num_batches if num_batches > 0 else 0
# #     for key in loss_components:
# #         loss_components[key] /= num_batches if num_batches > 0 else 1
    
# #     return avg_loss, loss_components


# # def main():
# #     parser = argparse.ArgumentParser(description='Train E2Net with DINOv2')
    
# #     # 数据参数
# #     parser.add_argument('--datapath', type=str, default='dataset/TrainDataset',
# #                         help='Path to training dataset')
# #     parser.add_argument('--val_datapath', type=str, default='dataset/TestDataset/CAMO',
# #                         help='Path to validation dataset')
    
# #     # 训练参数
# #     parser.add_argument('--batch_size', type=int, default=4, 
# #                         help='Batch size (reduce if OOM)')
# #     parser.add_argument('--epochs', type=int, default=100, 
# #                         help='Number of epochs')
# #     parser.add_argument('--lr', type=float, default=1e-4, 
# #                         help='Learning rate')
# #     parser.add_argument('--weight_decay', type=float, default=1e-4, 
# #                         help='Weight decay')
    
# #     # 模型参数
# #     parser.add_argument('--encoder_size', type=str, default='base',
# #                         choices=['small', 'base', 'large', 'giant'],
# #                         help='DINOv2 model size')
# #     parser.add_argument('--unified_channels', type=int, default=256, 
# #                         help='Unified channel dimension')
# #     parser.add_argument('--freeze_encoder', action='store_true', default=True,
# #                         help='Freeze DINOv2 encoder')
    
# #     # 损失权重
# #     parser.add_argument('--lambda_dice', type=float, default=1.0, 
# #                         help='Dice loss weight')
# #     parser.add_argument('--lambda_bce', type=float, default=1.0, 
# #                         help='BCE loss weight')
# #     parser.add_argument('--lambda_coarse', type=float, default=0.5, 
# #                         help='Coarse loss weight')
# #     parser.add_argument('--lambda_refined', type=float, default=0.3, 
# #                         help='Refined loss weight')
    
# #     # Checkpoint 参数
# #     parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/E2Net_DINOv2',
# #                         help='Directory to save checkpoints')
# #     parser.add_argument('--resume', type=str, default=None,
# #                         help='Path to checkpoint to resume from')
# #     parser.add_argument('--save_freq', type=int, default=10, 
# #                         help='Save checkpoint every N epochs')
    
# #     # Device
# #     parser.add_argument('--device', type=str, default='cuda',
# #                         help='Device to use (cuda or cpu)')
    
# #     # 图像尺寸
# #     parser.add_argument('--image_size', type=int, default=392,
# #                         help='Image size (must be multiple of 14 for DINOv2)')
    
# #     args = parser.parse_args()
    
# #     # 验证图像尺寸
# #     if args.image_size % 14 != 0:
# #         print(f"Warning: Image size {args.image_size} is not multiple of 14!")
# #         args.image_size = (args.image_size // 14) * 14
# #         print(f"Adjusted to {args.image_size}")
    
# #     # 创建 checkpoint 文件夹
# #     os.makedirs(args.checkpoint_dir, exist_ok=True)
    
# #     # Device
# #     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# #     print(f"\nUsing device: {device}")
# #     if device.type == 'cuda':
# #         print(f"GPU: {torch.cuda.get_device_name(0)}")
    
# #     # 加载数据集
# #     print("\nLoading datasets...")
    
# #     # 训练数据集
# #     cfg_train = Config(
# #         datapath=args.datapath,
# #         mode='train',
# #         snapshot=None,
# #         batch_size=args.batch_size
# #     )
# #     train_data = Data(cfg_train, 'E2Net')
# #     train_loader = DataLoader(
# #         train_data,
# #         batch_size=args.batch_size,
# #         shuffle=True,
# #         num_workers=4,
# #         pin_memory=True if device.type == 'cuda' else False,
# #         collate_fn=train_data.collate
# #     )
    
# #     # 验证数据集
# #     cfg_val = Config(
# #         datapath=args.val_datapath,
# #         mode='train',
# #         snapshot=None,
# #         batch_size=1
# #     )
# #     val_data = Data(cfg_val, 'E2Net')
# #     val_loader = DataLoader(
# #         val_data,
# #         batch_size=1,
# #         shuffle=False,
# #         num_workers=2,
# #         pin_memory=True if device.type == 'cuda' else False,
# #         collate_fn=val_data.collate
# #     )
    
# #     print(f"Training samples: {len(train_data)}")
# #     print(f"Validation samples: {len(val_data)}")
    
# #     # 创建模型
# #     print("\nCreating E2Net with DINOv2...")
# #     model = E2Net_DINOv2(
# #         encoder_size=args.encoder_size,
# #         freeze_encoder=args.freeze_encoder,
# #         unified_channels=args.unified_channels
# #     )
# #     model = model.to(device)
    
# #     # 统计参数
# #     total_params = sum(p.numel() for p in model.parameters())
# #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"\nModel Parameters:")
# #     print(f"  Total: {total_params:,}")
# #     print(f"  Trainable: {trainable_params:,}")
# #     print(f"  Frozen: {total_params - trainable_params:,}")
    
# #     # 优化器（仅优化可训练参数）
# #     optimizer = torch.optim.AdamW(
# #         filter(lambda p: p.requires_grad, model.parameters()),
# #         lr=args.lr,
# #         weight_decay=args.weight_decay
# #     )
    
# #     # 学习率调度器（余弦退火）
# #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
# #         optimizer,
# #         T_max=args.epochs,
# #         eta_min=1e-6
# #     )
    
# #     # 从检查点恢复训练
# #     start_epoch = 0
# #     best_val_loss = float('inf')
    
# #     if args.resume:
# #         print(f"\nResuming from checkpoint: {args.resume}")
# #         checkpoint = torch.load(args.resume, map_location=device)
# #         model.load_state_dict(checkpoint['model_state_dict'])
# #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# #         start_epoch = checkpoint['epoch'] + 1
# #         best_val_loss = checkpoint.get('best_val_loss', float('inf'))
# #         print(f"Resumed from epoch {start_epoch}")
    
# #     # 开始训练
# #     print("\n" + "="*70)
# #     print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# #     print("="*70)
    
# #     for epoch in range(start_epoch, args.epochs):
# #         print(f"\n{'='*70}")
# #         print(f"Epoch {epoch+1}/{args.epochs}")
# #         print(f"{'='*70}")
        
# #         # 训练
# #         train_loss, train_loss_components = train_epoch(
# #             model, train_loader, optimizer, device, epoch, args
# #         )
        
# #         print(f"\nTrain Loss: {train_loss:.4f}")
# #         print("  Components:")
# #         for key, value in train_loss_components.items():
# #             if key != 'total':
# #                 print(f"    {key}: {value:.4f}")
        
# #         # 验证
# #         val_loss, val_loss_components = validate(model, val_loader, device, args)
# #         print(f"\nValidation Loss: {val_loss:.4f}")
# #         print("  Components:")
# #         for key, value in val_loss_components.items():
# #             if key != 'total':
# #                 print(f"    {key}: {value:.4f}")
        
# #         # 更新学习率
# #         scheduler.step()
# #         current_lr = optimizer.param_groups[0]['lr']
# #         print(f"\nLearning rate: {current_lr:.6f}")
        
# #         # 保存 checkpoint
# #         if (epoch + 1) % args.save_freq == 0:
# #             checkpoint_path = os.path.join(
# #                 args.checkpoint_dir,
# #                 f'E2Net_DINOv2_epoch_{epoch+1}.pth'
# #             )
# #             torch.save({
# #                 'epoch': epoch,
# #                 'model_state_dict': model.state_dict(),
# #                 'optimizer_state_dict': optimizer.state_dict(),
# #                 'train_loss': train_loss,
# #                 'val_loss': val_loss,
# #                 'best_val_loss': best_val_loss,
# #                 'args': args
# #             }, checkpoint_path)
# #             print(f"\nCheckpoint saved: {checkpoint_path}")
        
# #         # 保存 best model
# #         if val_loss < best_val_loss:
# #             best_val_loss = val_loss
# #             best_model_path = os.path.join(args.checkpoint_dir, 'E2Net_DINOv2_best.pth')
# #             torch.save({
# #                 'epoch': epoch,
# #                 'model_state_dict': model.state_dict(),
# #                 'optimizer_state_dict': optimizer.state_dict(),
# #                 'train_loss': train_loss,
# #                 'val_loss': val_loss,
# #                 'best_val_loss': best_val_loss,
# #                 'args': args
# #             }, best_model_path)
# #             print(f"✓ Best model updated: {best_model_path} (val_loss: {val_loss:.4f})")
    
# #     print("\n" + "="*70)
# #     print(f"Training Completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# #     print(f"Best validation loss: {best_val_loss:.4f}")
# #     print("="*70)


# # if __name__ == '__main__':
# #     main()




# #v6
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 使用 DINOv2 编码器训练 E2Net 模型

# 改进项（相比原版）：
#   方案 B — 边界感知损失（Edge-aware Loss）
#            用 Sobel 算子从 GT mask 提取边界区域，对边界像素施加额外的
#            BCE + Dice 监督，迫使模型在细腻纹理边界处给出更精确的预测。
#            对 CHAMELEON 这类边界模糊的自然伪装场景效果显著。

#   方案 C — Adapter scale 自适应初始化
#            根据命令行参数 --adapter_scale 控制 FeatureAdapter 的残差
#            初始缩放因子。小数据集建议用更小的值（如 1e-4）防止过拟合，
#            大数据集可适当增大（如 1e-2）加快适配速度。
# """

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime

# from E2Net_dinov2 import E2Net_DINOv2
# from dataset import Data, Config


# # ═════════════════════════════════════════════════════════════════════════════
# # 损失函数
# # ═════════════════════════════════════════════════════════════════════════════

# def dice_loss(pred, target, smooth=1.0):
#     """Dice 损失，pred/target: [B, 1, H, W]"""
#     pred   = pred.contiguous().view(pred.size(0), -1)
#     target = target.contiguous().view(target.size(0), -1)
#     inter  = (pred * target).sum(dim=1)
#     union  = pred.sum(dim=1) + target.sum(dim=1)
#     return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


# def bce_loss(pred, target):
#     """二值交叉熵损失"""
#     return F.binary_cross_entropy(pred, target, reduction='mean')


# # ─────────────────────────────────────────────────────────────────────────────
# # 方案 B：边界感知损失
# # ─────────────────────────────────────────────────────────────────────────────

# def get_edge_mask(mask: torch.Tensor, dilation: int = 3) -> torch.Tensor:
#     """
#     用 Sobel 算子从二值 GT mask 提取边界区域，再膨胀 dilation 像素，
#     生成边界感知权重图（0/1）。

#     步骤：
#         1. Sobel-x + Sobel-y → 梯度幅值图
#         2. 阈值二值化（梯度 > 0 的像素为边界）
#         3. MaxPool 膨胀，扩大边界区域宽度，覆盖模糊边界的不确定像素

#     参数:
#         mask     : [B, 1, H, W]，值域 [0, 1] 的 GT mask
#         dilation : 边界膨胀像素数，默认 3
#     返回:
#         edge_mask : [B, 1, H, W]，边界区域为 1，内部/背景为 0
#     """
#     # Sobel 卷积核（在 mask 通道上逐样本计算）
#     sobel_x = torch.tensor(
#         [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#         dtype=mask.dtype, device=mask.device
#     ).view(1, 1, 3, 3)

#     sobel_y = torch.tensor(
#         [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#         dtype=mask.dtype, device=mask.device
#     ).view(1, 1, 3, 3)

#     # 计算梯度幅值
#     grad_x = F.conv2d(mask, sobel_x, padding=1)
#     grad_y = F.conv2d(mask, sobel_y, padding=1)
#     grad   = (grad_x ** 2 + grad_y ** 2).sqrt()

#     # 二值化：梯度 > 阈值处为边界
#     edge = (grad > 0.1).float()

#     # MaxPool 膨胀：kernel_size = 2*dilation+1，保持尺寸不变
#     if dilation > 0:
#         k    = 2 * dilation + 1
#         edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=dilation)

#     return edge   # [B, 1, H, W]


# def edge_aware_loss(pred: torch.Tensor,
#                     target: torch.Tensor,
#                     edge_mask: torch.Tensor) -> torch.Tensor:
#     """
#     边界感知损失：在边界区域施加额外的 BCE + Dice 监督。

#     仅对 edge_mask=1 的像素计算损失，迫使模型对模糊边界更精确。

#     参数:
#         pred      : [B, 1, H, W] 预测概率图
#         target    : [B, 1, H, W] GT mask（0/1）
#         edge_mask : [B, 1, H, W] 边界区域掩码（由 get_edge_mask 生成）
#     返回:
#         loss : 标量
#     """
#     # 提取边界像素（masked select 保持梯度）
#     pred_edge   = pred   * edge_mask
#     target_edge = target * edge_mask

#     n_edge = edge_mask.sum().clamp(min=1)

#     # 边界区域 BCE
#     bce_e = F.binary_cross_entropy(
#         pred_edge, target_edge, reduction='sum'
#     ) / n_edge

#     # 边界区域 Dice
#     inter   = (pred_edge * target_edge).sum()
#     union   = pred_edge.sum() + target_edge.sum()
#     dice_e  = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

#     return bce_e + dice_e


# def compute_loss(
#     predictions, masks,
#     lambda_dice=1.0, lambda_bce=1.0,
#     lambda_coarse=0.5, lambda_refined=0.3,
#     # 方案 B 参数
#     lambda_edge=0.3, edge_dilation=3,
# ):
#     """
#     计算 E2Net 的总损失（含多阶段监督 + 方案B边界感知损失）

#     参数:
#         predictions   : (Y_coarse, Y_refined, Y_final)
#         masks         : [B, 1, H, W] GT mask
#         lambda_edge   : 边界感知损失权重（0 表示关闭，推荐 0.2~0.5）
#         edge_dilation : 边界区域膨胀像素数
#     返回:
#         total_loss : 标量
#         loss_dict  : 各子损失字典（用于日志）
#     """
#     Y_coarse, Y_refined, Y_final = predictions

#     # 归一化到 [0, 1]
#     masks = masks / 255.0 if masks.max() > 1.0 else masks

#     # ── 主损失：最终预测 ───────────────────────────────────────────────────
#     loss_dice_final = dice_loss(Y_final, masks)
#     loss_bce_final  = bce_loss(Y_final,  masks)

#     # ── 辅助损失：粗略预测 ─────────────────────────────────────────────────
#     loss_dice_coarse = dice_loss(Y_coarse, masks)
#     loss_bce_coarse  = bce_loss(Y_coarse,  masks)

#     # ── 辅助损失：细化预测 ─────────────────────────────────────────────────
#     loss_dice_refined = dice_loss(Y_refined, masks)
#     loss_bce_refined  = bce_loss(Y_refined,  masks)

#     # ── 方案 B：边界感知损失（仅作用于最终预测） ───────────────────────────
#     loss_edge = torch.tensor(0.0, device=masks.device)
#     if lambda_edge > 0:
#         edge_mask  = get_edge_mask(masks, dilation=edge_dilation)
#         loss_edge  = edge_aware_loss(Y_final, masks, edge_mask)

#     # ── 加权求和 ───────────────────────────────────────────────────────────
#     loss_final   = lambda_dice * loss_dice_final + lambda_bce * loss_bce_final
#     loss_coarse  = lambda_coarse  * (loss_dice_coarse  + loss_bce_coarse)
#     loss_refined = lambda_refined * (loss_dice_refined + loss_bce_refined)
#     loss_edge_w  = lambda_edge * loss_edge

#     total_loss = loss_final + loss_coarse + loss_refined + loss_edge_w

#     loss_dict = {
#         'total'       : total_loss.item(),
#         'dice_final'  : loss_dice_final.item(),
#         'bce_final'   : loss_bce_final.item(),
#         'dice_coarse' : loss_dice_coarse.item(),
#         'bce_coarse'  : loss_bce_coarse.item(),
#         'dice_refined': loss_dice_refined.item(),
#         'bce_refined' : loss_bce_refined.item(),
#         'edge'        : loss_edge.item(),        # 方案 B 监控项
#     }

#     return total_loss, loss_dict


# # ═════════════════════════════════════════════════════════════════════════════
# # 训练 / 验证循环
# # ═════════════════════════════════════════════════════════════════════════════

# def train_epoch(model, dataloader, optimizer, device, epoch, args):
#     model.train()

#     epoch_loss      = 0.0
#     loss_components = {k: 0.0 for k in [
#         'total', 'dice_final', 'bce_final',
#         'dice_coarse', 'bce_coarse',
#         'dice_refined', 'bce_refined', 'edge'
#     ]}

#     pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')

#     for images, masks in pbar:
#         images = images.to(device, dtype=torch.float32)
#         masks  = masks.to(device,  dtype=torch.float32)

#         predictions = model(images)

#         loss, loss_dict = compute_loss(
#             predictions, masks,
#             lambda_dice    = args.lambda_dice,
#             lambda_bce     = args.lambda_bce,
#             lambda_coarse  = args.lambda_coarse,
#             lambda_refined = args.lambda_refined,
#             lambda_edge    = args.lambda_edge,       # 方案 B
#             edge_dilation  = args.edge_dilation,     # 方案 B
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         epoch_loss += loss.item()
#         for key in loss_components:
#             if key in loss_dict:
#                 loss_components[key] += loss_dict[key]

#         pbar.set_postfix({
#             'loss' : f"{loss.item():.4f}",
#             'dice' : f"{loss_dict['dice_final']:.4f}",
#             'edge' : f"{loss_dict['edge']:.4f}",
#         })

#     n = len(dataloader)
#     epoch_loss /= n
#     for key in loss_components:
#         loss_components[key] /= n

#     return epoch_loss, loss_components


# def validate(model, dataloader, device, args):
#     model.eval()

#     total_loss      = 0.0
#     loss_components = {'total': 0.0, 'dice_final': 0.0, 'bce_final': 0.0, 'edge': 0.0}

#     with torch.no_grad():
#         pbar = tqdm(dataloader, desc='Validation')

#         for images, masks in pbar:
#             images = images.to(device, dtype=torch.float32)
#             masks  = masks.to(device,  dtype=torch.float32)

#             predictions = model(images)

#             loss, loss_dict = compute_loss(
#                 predictions, masks,
#                 lambda_dice    = args.lambda_dice,
#                 lambda_bce     = args.lambda_bce,
#                 lambda_coarse  = args.lambda_coarse,
#                 lambda_refined = args.lambda_refined,
#                 lambda_edge    = args.lambda_edge,
#                 edge_dilation  = args.edge_dilation,
#             )

#             total_loss += loss.item()
#             for key in ['dice_final', 'bce_final', 'edge']:
#                 if key in loss_dict:
#                     loss_components[key] += loss_dict[key]

#             pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

#     n = max(len(dataloader), 1)
#     avg_loss = total_loss / n
#     for key in loss_components:
#         loss_components[key] /= n

#     return avg_loss, loss_components


# # ═════════════════════════════════════════════════════════════════════════════
# # 主入口
# # ═════════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(description='Train E2Net with DINOv2 (方案B+C)')

#     # ── 数据 ──────────────────────────────────────────────────────────────
#     parser.add_argument('--datapath',     type=str, default='dataset/TrainDataset')
#     parser.add_argument('--val_datapath', type=str, default='dataset/TestDataset/CAMO')

#     # ── 训练 ──────────────────────────────────────────────────────────────
#     parser.add_argument('--batch_size',   type=int,   default=4)
#     parser.add_argument('--epochs',       type=int,   default=100)
#     parser.add_argument('--lr',           type=float, default=1e-4)
#     parser.add_argument('--weight_decay', type=float, default=1e-4)
#     parser.add_argument('--image_size',   type=int,   default=392)

#     # ── 模型 ──────────────────────────────────────────────────────────────
#     parser.add_argument('--encoder_size',     type=str,  default='base',
#                         choices=['small', 'base', 'large', 'giant'])
#     parser.add_argument('--unified_channels', type=int,  default=256)
#     parser.add_argument('--freeze_encoder',   action='store_true', default=True)

#     # ── Adapter 参数（方案 C）─────────────────────────────────────────────
#     parser.add_argument('--adapter_at', type=int, nargs='+', default=[3, 6, 9, 11],
#                         help='插入 Parallel Adapter 的层索引（0-based）')
#     parser.add_argument('--adapter_reduction', type=int, default=4,
#                         help='Adapter 瓶颈压缩比')
#     parser.add_argument(
#         '--adapter_scale', type=float, default=1e-3,
#         help=(
#             '【方案 C】Adapter 残差初始缩放因子。\n'
#             '  小数据集（如 CHAMELEON ~76张）建议 1e-4，防止过拟合；\n'
#             '  大数据集（如 COD10K）可用 1e-2，加快适配速度。\n'
#             '  默认 1e-3（折中）。'
#         )
#     )

#     # ── 基础损失权重 ───────────────────────────────────────────────────────
#     parser.add_argument('--lambda_dice',    type=float, default=1.0)
#     parser.add_argument('--lambda_bce',     type=float, default=1.0)
#     parser.add_argument('--lambda_coarse',  type=float, default=0.5)
#     parser.add_argument('--lambda_refined', type=float, default=0.3)

#     # ── 方案 B：边界感知损失 ───────────────────────────────────────────────
#     parser.add_argument(
#         '--lambda_edge', type=float, default=0.3,
#         help=(
#             '【方案 B】边界感知损失权重。\n'
#             '  0 = 关闭（等价于原版）；推荐范围 0.2~0.5。\n'
#             '  较大值使模型更关注边界，适合 CHAMELEON 细腻纹理场景。\n'
#             '  默认 0.3。'
#         )
#     )
#     parser.add_argument(
#         '--edge_dilation', type=int, default=3,
#         help=(
#             '【方案 B】边界区域膨胀像素数（覆盖模糊边界不确定区域）。\n'
#             '  默认 3，值越大边界区域越宽。'
#         )
#     )

#     # ── Checkpoint ────────────────────────────────────────────────────────
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/E2Net_DINOv2')
#     parser.add_argument('--resume',         type=str, default=None)
#     parser.add_argument('--save_freq',      type=int, default=10)

#     # ── Device ────────────────────────────────────────────────────────────
#     parser.add_argument('--device', type=str, default='cuda')

#     args = parser.parse_args()

#     # 验证图像尺寸（DINOv2 要求 14 的倍数）
#     if args.image_size % 14 != 0:
#         args.image_size = (args.image_size // 14) * 14
#         print(f"Image size adjusted to {args.image_size}")

#     os.makedirs(args.checkpoint_dir, exist_ok=True)

#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print(f"\nUsing device: {device}")
#     if device.type == 'cuda':
#         print(f"GPU: {torch.cuda.get_device_name(0)}")

#     # ── 打印改进配置 ───────────────────────────────────────────────────────
#     print("\n" + "=" * 60)
#     print("改进配置")
#     print("=" * 60)
#     print(f"【方案 B】边界感知损失")
#     print(f"  lambda_edge    = {args.lambda_edge}  "
#           f"({'启用' if args.lambda_edge > 0 else '关闭'})")
#     print(f"  edge_dilation  = {args.edge_dilation} px")
#     print(f"【方案 C】Adapter scale 自适应")
#     print(f"  adapter_at     = {args.adapter_at}")
#     print(f"  adapter_scale  = {args.adapter_scale}  "
#           f"({'保守，适合小数据' if args.adapter_scale < 1e-2 else '积极，适合大数据'})")
#     print(f"  adapter_reduction = {args.adapter_reduction}")
#     print("=" * 60)

#     # ── 数据集 ─────────────────────────────────────────────────────────────
#     print("\nLoading datasets...")
#     cfg_train = Config(datapath=args.datapath,     mode='train',
#                        snapshot=None, batch_size=args.batch_size)
#     cfg_val   = Config(datapath=args.val_datapath, mode='train',
#                        snapshot=None, batch_size=1)

#     train_data   = Data(cfg_train, 'E2Net')
#     val_data     = Data(cfg_val,   'E2Net')

#     train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
#                               num_workers=4,
#                               pin_memory=(device.type == 'cuda'),
#                               collate_fn=train_data.collate)
#     val_loader   = DataLoader(val_data,   batch_size=1, shuffle=False,
#                               num_workers=2,
#                               pin_memory=(device.type == 'cuda'),
#                               collate_fn=val_data.collate)

#     print(f"Training samples  : {len(train_data)}")
#     print(f"Validation samples: {len(val_data)}")

#     # ── 创建模型（传入方案 C 的 adapter_scale） ────────────────────────────
#     print("\nCreating E2Net with DINOv2...")
#     model = E2Net_DINOv2(
#         encoder_size      = args.encoder_size,
#         freeze_encoder    = args.freeze_encoder,
#         unified_channels  = args.unified_channels,
#         adapter_at        = args.adapter_at,
#         adapter_reduction = args.adapter_reduction,
#         adapter_scale     = args.adapter_scale,     # 方案 C
#     )
#     model = model.to(device)

#     total_params     = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nModel Parameters:")
#     print(f"  Total    : {total_params:,}")
#     print(f"  Trainable: {trainable_params:,}")
#     print(f"  Frozen   : {total_params - trainable_params:,}")

#     # ── 优化器 / 调度器 ───────────────────────────────────────────────────
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.lr, weight_decay=args.weight_decay
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=args.epochs, eta_min=1e-6
#     )

#     # ── 断点续训 ──────────────────────────────────────────────────────────
#     start_epoch   = 0
#     best_val_loss = float('inf')

#     if args.resume:
#         print(f"\nResuming from: {args.resume}")
#         ckpt = torch.load(args.resume, map_location=device)
#         model.load_state_dict(ckpt['model_state_dict'])
#         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#         start_epoch   = ckpt['epoch'] + 1
#         best_val_loss = ckpt.get('best_val_loss', float('inf'))
#         print(f"Resumed from epoch {start_epoch}")

#     # ── 训练主循环 ────────────────────────────────────────────────────────
#     print("\n" + "=" * 70)
#     print(f"Training Start — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("=" * 70)

#     for epoch in range(start_epoch, args.epochs):
#         print(f"\n{'='*70}\nEpoch {epoch+1}/{args.epochs}\n{'='*70}")

#         # 训练
#         train_loss, train_components = train_epoch(
#             model, train_loader, optimizer, device, epoch, args
#         )
#         print(f"\nTrain Loss: {train_loss:.4f}")
#         for k, v in train_components.items():
#             if k != 'total':
#                 print(f"  {k:<16}: {v:.4f}")

#         # 验证
#         val_loss, val_components = validate(model, val_loader, device, args)
#         print(f"\nValidation Loss: {val_loss:.4f}")
#         for k, v in val_components.items():
#             if k != 'total':
#                 print(f"  {k:<16}: {v:.4f}")

#         scheduler.step()
#         print(f"\nLR: {optimizer.param_groups[0]['lr']:.6f}")

#         # 周期性保存
#         if (epoch + 1) % args.save_freq == 0:
#             ckpt_path = os.path.join(
#                 args.checkpoint_dir, f'E2Net_DINOv2_epoch_{epoch+1}.pth'
#             )
#             torch.save({
#                 'epoch'              : epoch,
#                 'model_state_dict'   : model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss'         : train_loss,
#                 'val_loss'           : val_loss,
#                 'best_val_loss'      : best_val_loss,
#                 'args'               : args,
#             }, ckpt_path)
#             print(f"\nCheckpoint saved: {ckpt_path}")

#         # 保存最优模型
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_path     = os.path.join(args.checkpoint_dir, 'E2Net_DINOv2_best.pth')
#             torch.save({
#                 'epoch'              : epoch,
#                 'model_state_dict'   : model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss'         : train_loss,
#                 'val_loss'           : val_loss,
#                 'best_val_loss'      : best_val_loss,
#                 'args'               : args,
#             }, best_path)
#             print(f"✓ Best model → {best_path}  (val_loss={val_loss:.4f})")

#     print("\n" + "=" * 70)
#     print(f"Training Completed — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Best validation loss: {best_val_loss:.4f}")
#     print("=" * 70)


# if __name__ == '__main__':
#     main()




#v7
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 DINOv2 编码器训练 E2Net 模型

改进项：
  优化 A — 自适应边界感知损失（Adaptive Edge-aware Loss）

  问题回顾：
    原方案 B 对所有图像施加相同的边界损失权重 lambda_edge，
    导致 CHAMELEON（自然动物，边界极度模糊）上性能退化：
    Sobel 提取的"边界"本身是噪声，强监督反而惩罚了合理预测。

  优化 A 的修复方案：
    根据 GT mask 的 Sobel 梯度幅值标准差（clarity）
    自动衡量当前 batch 边界的清晰程度：

        clarity      = std( Sobel梯度幅值图 )
        adaptive_λ   = lambda_edge × sigmoid( 20 × (clarity − 0.05) )

    边界清晰（CAMO/COD10K，硬边界）→ clarity 大 → λ 接近 lambda_edge
    边界模糊（CHAMELEON，自然纹理）→ clarity 小 → λ 自动趋近 0

    结果：在不改动任何模型结构的情况下，清晰边界场景获得边界监督增益，
         模糊边界场景不受额外惩罚。

  注意：本文件只修改损失函数部分，模型接口与原版完全兼容（返回三个预测）。
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

from E2Net_dinov2 import E2Net_DINOv2
from dataset import Data, Config


# ═════════════════════════════════════════════════════════════════════════════
# 基础损失函数（与原版相同）
# ═════════════════════════════════════════════════════════════════════════════

def dice_loss(pred, target, smooth=1.0):
    """Dice 损失，pred/target: [B, 1, H, W]"""
    pred   = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    union  = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def bce_loss(pred, target):
    """二值交叉熵损失"""
    return F.binary_cross_entropy(pred, target, reduction='mean')


# ═════════════════════════════════════════════════════════════════════════════
# 优化 A：自适应边界感知损失
# ═════════════════════════════════════════════════════════════════════════════

def get_edge_mask(mask: torch.Tensor, dilation: int = 3) -> torch.Tensor:
    """
    用 Sobel 算子从二值 GT mask 提取边界区域，再 MaxPool 膨胀。

    参数:
        mask     : [B, 1, H, W]，值域 [0, 1]
        dilation : 边界膨胀像素数
    返回:
        edge_mask : [B, 1, H, W]，边界区域为 1
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=mask.dtype, device=mask.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=mask.dtype, device=mask.device
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)
    grad   = (grad_x ** 2 + grad_y ** 2).sqrt()

    edge = (grad > 0.1).float()
    if dilation > 0:
        k    = 2 * dilation + 1
        edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=dilation)
    return edge


def adaptive_edge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_edge: float,
    edge_dilation: int = 3,
) -> tuple:
    """
    自适应边界感知损失（优化 A 核心函数）

    自适应权重计算：
        1. 用 Sobel 算子计算 GT mask 的梯度幅值图
        2. 取幅值图标准差作为边界清晰度指标（clarity）
        3. 用 sigmoid 门控将 clarity 映射到 [0, lambda_edge]：
               adaptive_λ = lambda_edge × sigmoid(20 × (clarity − 0.05))
           参数含义：
               k=20         控制门控的陡峭程度（越大过渡越锐利）
               threshold=0.05 为清晰/模糊边界的分界点
               clarity < 0.05 → sigmoid ≈ 0 → λ ≈ 0（不惩罚模糊边界）
               clarity > 0.10 → sigmoid ≈ 1 → λ ≈ lambda_edge（完整监督）

    边界区域损失 = BCE(pred·mask_e, target·mask_e) + Dice(pred·mask_e, target·mask_e)

    参数:
        pred          : [B, 1, H, W] 预测概率图
        target        : [B, 1, H, W] GT mask（0/1）
        lambda_edge   : 边界损失权重上限
        edge_dilation : 边界区域膨胀像素数

    返回:
        loss            : 加权边界损失（标量 Tensor）
        adaptive_lambda : 本次实际使用的 λ 值（float，供日志监控）
    """
    # ── Step 1：计算梯度幅值图（复用 sobel） ──────────────────────────────
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=target.dtype, device=target.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=target.dtype, device=target.device
    ).view(1, 1, 3, 3)
    grad = (F.conv2d(target, sobel_x, padding=1) ** 2
          + F.conv2d(target, sobel_y, padding=1) ** 2).sqrt()  # [B,1,H,W]

    # ── Step 2：边界清晰度（batch 内全图梯度幅值标准差） ──────────────────
    clarity = grad.std().clamp(min=1e-6)   # 标量

    # ── Step 3：自适应 λ ──────────────────────────────────────────────────
    k, threshold    = 20.0, 0.05
    adaptive_lambda = lambda_edge * torch.sigmoid(
        torch.tensor(
            k * (clarity.item() - threshold),
            dtype=torch.float32, device=target.device
        )
    )

    # ── Step 4：提取边界掩码并计算损失 ────────────────────────────────────
    edge_mask = get_edge_mask(target, dilation=edge_dilation)  # [B,1,H,W]
    n_edge    = edge_mask.sum().clamp(min=1)

    pred_e   = pred   * edge_mask
    target_e = target * edge_mask

    # 边界区域 BCE
    bce_e = F.binary_cross_entropy(pred_e, target_e, reduction='sum') / n_edge

    # 边界区域 Dice
    inter  = (pred_e * target_e).sum()
    union  = pred_e.sum() + target_e.sum()
    dice_e = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

    loss = adaptive_lambda * (bce_e + dice_e)

    return loss, adaptive_lambda.item()


# ═════════════════════════════════════════════════════════════════════════════
# 总损失（在原版基础上加入优化 A）
# ═════════════════════════════════════════════════════════════════════════════

def compute_loss(
    predictions,
    masks,
    lambda_dice    = 1.0,
    lambda_bce     = 1.0,
    lambda_coarse  = 0.5,
    lambda_refined = 0.3,
    lambda_edge    = 0.3,   # 优化 A：边界损失权重上限（0 = 关闭）
    edge_dilation  = 3,     # 优化 A：边界膨胀像素数
):
    """
    计算 E2Net 总损失（含优化 A 自适应边界损失）

    总损失 = loss_final + loss_coarse + loss_refined
           + adaptive_λ_edge × edge_loss     ← 优化 A

    与原版的差别仅在最后一项；设 lambda_edge=0 可完全还原原版损失。
    """
    Y_coarse, Y_refined, Y_final = predictions
    masks = masks / 255.0 if masks.max() > 1.0 else masks

    # ── 原版三项损失（不变）─────────────────────────────────────────────────
    loss_dice_final   = dice_loss(Y_final,   masks)
    loss_bce_final    = bce_loss(Y_final,    masks)
    loss_dice_coarse  = dice_loss(Y_coarse,  masks)
    loss_bce_coarse   = bce_loss(Y_coarse,   masks)
    loss_dice_refined = dice_loss(Y_refined, masks)
    loss_bce_refined  = bce_loss(Y_refined,  masks)

    loss_final   = lambda_dice    * loss_dice_final + lambda_bce * loss_bce_final
    loss_coarse  = lambda_coarse  * (loss_dice_coarse  + loss_bce_coarse)
    loss_refined = lambda_refined * (loss_dice_refined + loss_bce_refined)

    # ── 优化 A：自适应边界损失 ─────────────────────────────────────────────
    loss_edge      = torch.tensor(0.0, device=masks.device)
    adaptive_lam_e = 0.0
    if lambda_edge > 0:
        loss_edge, adaptive_lam_e = adaptive_edge_loss(
            Y_final, masks, lambda_edge, edge_dilation
        )

    total_loss = loss_final + loss_coarse + loss_refined + loss_edge

    loss_dict = {
        'total'          : total_loss.item(),
        'dice_final'     : loss_dice_final.item(),
        'bce_final'      : loss_bce_final.item(),
        'dice_coarse'    : loss_dice_coarse.item(),
        'bce_coarse'     : loss_bce_coarse.item(),
        'dice_refined'   : loss_dice_refined.item(),
        'bce_refined'    : loss_bce_refined.item(),
        'edge'           : loss_edge.item() if torch.is_tensor(loss_edge) else loss_edge,
        'adaptive_lam_e' : adaptive_lam_e,   # 监控：实际使用的 λ
    }
    return total_loss, loss_dict


# ═════════════════════════════════════════════════════════════════════════════
# 训练 / 验证循环
# ═════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, device, epoch, args):
    model.train()
    epoch_loss      = 0.0
    loss_components = {k: 0.0 for k in [
        'total', 'dice_final', 'bce_final',
        'dice_coarse', 'bce_coarse',
        'dice_refined', 'bce_refined',
        'edge', 'adaptive_lam_e',
    ]}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')

    for images, masks in pbar:
        images = images.to(device, dtype=torch.float32)
        masks  = masks.to(device,  dtype=torch.float32)

        # 模型接口与原版相同，返回三个预测
        predictions = model(images)

        loss, loss_dict = compute_loss(
            predictions, masks,
            lambda_dice    = args.lambda_dice,
            lambda_bce     = args.lambda_bce,
            lambda_coarse  = args.lambda_coarse,
            lambda_refined = args.lambda_refined,
            lambda_edge    = args.lambda_edge,
            edge_dilation  = args.edge_dilation,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]

        pbar.set_postfix({
            'loss'  : f"{loss.item():.4f}",
            'edge_λ': f"{loss_dict['adaptive_lam_e']:.3f}",  # 监控自适应权重
            'edge'  : f"{loss_dict['edge']:.4f}",
        })

    n = len(dataloader)
    return epoch_loss / n, {k: v / n for k, v in loss_components.items()}


def validate(model, dataloader, device, args):
    model.eval()
    total_loss      = 0.0
    loss_components = {k: 0.0 for k in
                       ['total', 'dice_final', 'bce_final', 'edge', 'adaptive_lam_e']}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device, dtype=torch.float32)
            masks  = masks.to(device,  dtype=torch.float32)

            predictions = model(images)
            loss, loss_dict = compute_loss(
                predictions, masks,
                lambda_dice    = args.lambda_dice,
                lambda_bce     = args.lambda_bce,
                lambda_coarse  = args.lambda_coarse,
                lambda_refined = args.lambda_refined,
                lambda_edge    = args.lambda_edge,
                edge_dilation  = args.edge_dilation,
            )

            total_loss += loss.item()
            for key in ['dice_final', 'bce_final', 'edge', 'adaptive_lam_e']:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]

            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    n = max(len(dataloader), 1)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train E2Net（优化A：自适应边界损失）')

    # ── 数据 ──────────────────────────────────────────────────────────────
    parser.add_argument('--datapath',         type=str,   default='dataset/TrainDataset')
    parser.add_argument('--val_datapath',     type=str,   default='dataset/TestDataset/CAMO')

    # ── 训练 ──────────────────────────────────────────────────────────────
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--epochs',           type=int,   default=100)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--weight_decay',     type=float, default=1e-4)
    parser.add_argument('--image_size',       type=int,   default=392)

    # ── 模型（与原版相同）────────────────────────────────────────────────
    parser.add_argument('--encoder_size',     type=str,   default='base',
                        choices=['small', 'base', 'large', 'giant'])
    parser.add_argument('--unified_channels', type=int,   default=256)
    parser.add_argument('--freeze_encoder',   action='store_true', default=True)
    parser.add_argument('--adapter_at',       type=int,   nargs='+', default=[3,6,9,11])
    parser.add_argument('--adapter_reduction',type=int,   default=4)
    parser.add_argument('--adapter_scale',    type=float, default=1e-3)

    # ── 原版损失权重（不变）──────────────────────────────────────────────
    parser.add_argument('--lambda_dice',      type=float, default=1.0)
    parser.add_argument('--lambda_bce',       type=float, default=1.0)
    parser.add_argument('--lambda_coarse',    type=float, default=0.5)
    parser.add_argument('--lambda_refined',   type=float, default=0.3)

    # ── 优化 A：自适应边界损失 ─────────────────────────────────────────────
    parser.add_argument(
        '--lambda_edge', type=float, default=0.3,
        help=(
            '【优化A】边界损失权重上限。实际权重由 batch 边界清晰度自动缩放：\n'
            '  adaptive_λ = lambda_edge × sigmoid(20 × (clarity − 0.05))\n'
            '  清晰边界（CAMO/COD10K） → clarity 大 → λ 接近 lambda_edge\n'
            '  模糊边界（CHAMELEON）   → clarity 小 → λ 自动趋近 0\n'
            '  0 = 完全关闭（等价于原版损失）。推荐范围 0.2~0.4。'
        )
    )
    parser.add_argument(
        '--edge_dilation', type=int, default=3,
        help='【优化A】边界区域膨胀像素数，覆盖模糊边界的不确定区域。默认 3。'
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint_dir',   type=str,   default='checkpoint/E2Net_DINOv2')
    parser.add_argument('--resume',           type=str,   default=None)
    parser.add_argument('--save_freq',        type=int,   default=10)
    parser.add_argument('--device',           type=str,   default='cuda')

    args = parser.parse_args()

    # 验证图像尺寸
    if args.image_size % 14 != 0:
        args.image_size = (args.image_size // 14) * 14
        print(f"Image size adjusted to {args.image_size}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 打印优化 A 配置 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【优化 A】自适应边界感知损失")
    print(f"  lambda_edge 上限 = {args.lambda_edge}"
          f"  {'(已启用)' if args.lambda_edge > 0 else '(已关闭)'}")
    print(f"  edge_dilation    = {args.edge_dilation} px")
    print(f"  门控公式: adaptive_λ = {args.lambda_edge} × sigmoid(20×(clarity−0.05))")
    print(f"  → 模糊边界(CHAMELEON) clarity<0.05 时 λ→0，不施加强监督")
    print(f"  → 清晰边界(CAMO/COD10K) clarity>0.10 时 λ→{args.lambda_edge}，完整监督")
    print("=" * 60)

    # ── 数据集 ─────────────────────────────────────────────────────────────
    cfg_train = Config(datapath=args.datapath,     mode='train',
                       snapshot=None, batch_size=args.batch_size)
    cfg_val   = Config(datapath=args.val_datapath, mode='train',
                       snapshot=None, batch_size=1)
    train_data   = Data(cfg_train, 'E2Net')
    val_data     = Data(cfg_val,   'E2Net')
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == 'cuda'),
        collate_fn=train_data.collate
    )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
        collate_fn=val_data.collate
    )
    print(f"\nTrain: {len(train_data)} samples  Val: {len(val_data)} samples")

    # ── 创建模型（与原版接口完全一致）────────────────────────────────────
    print("\nCreating E2Net with DINOv2...")
    model = E2Net_DINOv2(
        encoder_size      = args.encoder_size,
        freeze_encoder    = args.freeze_encoder,
        unified_channels  = args.unified_channels,
        adapter_at        = args.adapter_at,
        adapter_reduction = args.adapter_reduction,
        adapter_scale     = args.adapter_scale,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total:{total:,}  Trainable:{trainable:,}  Frozen:{total-trainable:,}")

    # ── 优化器 / 调度器 ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── 断点续训 ──────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # ── 训练主循环 ────────────────────────────────────────────────────────
    print(f"\nTraining Start — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, tc = train_epoch(
            model, train_loader, optimizer, device, epoch, args
        )
        print(f"  Train Loss   : {train_loss:.4f}")
        print(f"  dice_final   : {tc['dice_final']:.4f}  "
              f"bce_final: {tc['bce_final']:.4f}")
        print(f"  edge_loss    : {tc['edge']:.4f}  "
              f"adaptive_λ(avg): {tc['adaptive_lam_e']:.4f}")

        val_loss, vc = validate(model, val_loader, device, args)
        print(f"  Val Loss     : {val_loss:.4f}  "
              f"dice={vc['dice_final']:.4f}  "
              f"edge={vc['edge']:.4f}  "
              f"adaptive_λ={vc['adaptive_lam_e']:.4f}")

        scheduler.step()
        print(f"  LR           : {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f'E2Net_DINOv2_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss,
                'best_val_loss': best_val_loss, 'args': args,
            }, ckpt_path)
            print(f"  Checkpoint   : {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'E2Net_DINOv2_best.pth')
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss,
                'best_val_loss': best_val_loss, 'args': args,
            }, best_path)
            print(f"  ✓ Best model → val_loss={val_loss:.4f}")

    print(f"\nCompleted — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()