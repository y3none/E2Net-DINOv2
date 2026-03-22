#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2Net + DINOv2 Encoder
"""

import torch
import torch.nn as nn
from dinov2_encoder_v4 import DINOv2Encoder
from typing import List

class E2Net_DINOv2(nn.Module):
    """
    基于 DINOv2 编码器的 E2Net 模型

    参数:
        encoder_size: DINOv2 模型尺寸，可选 'small'、'base'、'large'、'giant'
        freeze_encoder: 是否冻结 DINOv2 编码器的权重
        unified_channels: CAEM 模块输出的统一通道数
    """
    
    def __init__(
        self,
        encoder_size: str       = 'base',
        freeze_encoder: bool    = True,
        unified_channels: int   = 256,
        adapter_at: List[int]   = None,
        adapter_reduction: int  = 4,
        adapter_scale: float    = 1e-3, 
    ):
        super(E2Net_DINOv2, self).__init__()
        
        print("="*60)
        print("Initializing E2Net with DINOv2")
        print("="*60)
        
        # DINOv2 编码器（含 Parallel Adapter）
        self.encoder = DINOv2Encoder(
            model_size        = encoder_size,
            freeze            = freeze_encoder,
            pretrained        = True,
            adapter_at        = adapter_at or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            adapter_reduction = adapter_reduction,
            adapter_scale     = adapter_scale, 
        )
        
        # 获取编码器各阶段的输出通道数
        encoder_channels = self.encoder.out_channels
        print(f"Encoder channels: {encoder_channels}")
        
        # 导入其他模块
        from caem import CAEM
        from lfsm import LFSM
        from cfzm import CFZM
        from ccm import CCM
        
        # Channel Alignment and Enhancement Module
        self.caem = CAEM(
            in_channels=encoder_channels,
            unified_channels=unified_channels
        )
        
        # Lateral Fovea Scanning Module
        self.lfsm = LFSM(
            in_channels=unified_channels
        )
        
        # Central Fovea Zooming Module
        self.cfzm = CFZM(
            channels=unified_channels
        )
        
        # Cognitive Confirmation Module
        self.ccm = CCM(
            channels=unified_channels
        )
        
        # 最终预测头（三个阶段分别输出）
        self.pred_lfsm = nn.Sequential(
            nn.Conv2d(unified_channels, unified_channels // 2, 3, padding=1),
            nn.BatchNorm2d(unified_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_channels // 2, 1, 1)
        )
        
        self.pred_cfzm = nn.Sequential(
            nn.Conv2d(unified_channels, unified_channels // 2, 3, padding=1),
            nn.BatchNorm2d(unified_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_channels // 2, 1, 1)
        )
        
        self.pred_ccm = nn.Sequential(
            nn.Conv2d(unified_channels, unified_channels // 2, 3, padding=1),
            nn.BatchNorm2d(unified_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_channels // 2, 1, 1)
        )
        
        print("E2Net with DINOv2 initialized")
        print("="*60 + "\n")
    
    def forward(self, x):
        """
        前向传播过程

        参数:
            x: [B, 3, H, W] 输入图像

        返回:
            p_lfsm: [B, 1, H, W] LFSM 阶段的粗略预测
            p_cfzm: [B, 1, H, W] CFZM 阶段的细化预测  
            p_ccm: [B, 1, H, W] CCM 阶段的最终预测结果
        """
        B, C, H, W = x.shape
        
        # 特征提取（冻结）
        features = self.encoder(x)
        # features: [F1, F2, F3, F4]
        
        # 特征对齐增强
        unified_features = self.caem(features)
        # unified_features: [G1, G2, G3, G4]
        
        # Stage 1: 全局搜索
        # LFSM 以 G4（语义最强的高层特征）作为输入
        Y_coarse = self.lfsm(unified_features[3])  # G4
        
        # Stage 2: 动态聚焦
        # CFZM 接收统一后的特征列表和粗略预测图
        G_prime_features, Y_refined, Y_maps = self.cfzm(unified_features, Y_coarse)
        
        # Stage 3: 认知确认+分割
        # CCM 对增强后的特征进行最终融合
        Y_final = self.ccm(G_prime_features)
        
        # 上采样到输入分辨率
        Y_coarse = torch.nn.functional.interpolate(
            Y_coarse, size=(H, W), mode='bilinear', align_corners=False
        )
        Y_refined = torch.nn.functional.interpolate(
            Y_refined, size=(H, W), mode='bilinear', align_corners=False
        )
        Y_final = torch.nn.functional.interpolate(
            Y_final, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return Y_coarse, Y_refined, Y_final


if __name__ == '__main__':
    print("Testing E2Net with DINOv2\n")
    
    try:
        # 创建模型
        model = E2Net_DINOv2(
            encoder_size='base',
            freeze_encoder=True,
            unified_channels=256
        )
        
        # 测试前向传播，输入尺寸需为 14 的倍数
        x = torch.randn(2, 3, 392, 392)
        
        print("Running forward pass...")
        p_coarse, p_refined, p_final = model(x)
        
        print("\n✓ Forward pass successful!")
        print(f"  Coarse output: {p_coarse.shape}")
        print(f"  Refined output: {p_refined.shape}")
        print(f"  Final output: {p_final.shape}")
        
        # 统计参数量
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nParameters:")
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Frozen: {total - trainable:,}")
        
        print("\n✓ E2Net-DINOv2 ready for training!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()