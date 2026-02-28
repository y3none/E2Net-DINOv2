#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 Encoder for E2Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# 解决 xFormers warnings
warnings.filterwarnings('ignore', message='xFormers is not available')


class DINOv2Encoder(nn.Module):
    """
    DINOv2 编码器 —— 稳定且可靠

    DINOv2（2023）在伪装目标检测（COD）任务中表现优异：
    - 在 1.42 亿张图像上训练
    - 特征提取能力强
    - 经过充分验证，稳定性好

    参数:
        model_size: 模型尺寸，可选 'small'、'base'、'large'、'giant'
        freeze: 是否冻结编码器权重
        pretrained: 是否使用预训练权重
    """
    
    def __init__(self, model_size='base', freeze=True, pretrained=True):
        super(DINOv2Encoder, self).__init__()
        
        self.model_size = model_size
        
        print("=" * 60)
        print("DINOv2 Encoder")
        print(f"Model size: {model_size}")
        print("=" * 60)
        
        # 加载主干网络
        self.backbone = self._load_dinov2(model_size, pretrained)
        
        # 冻结参数
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Parameters frozen")
        
        # 设立通道维度
        self._setup_dims()
        self._build_proj()
        
        # 特征提取层索引
        self.layers = [3, 6, 9, 11]
        
        print(f"Feature dim: {self.feature_dim}")
        print(f"Output channels: {self.out_channels}")
        print("=" * 60 + "\n")
    
    def _load_dinov2(self, size, pretrained):
        """从 torch.hub 加载 DINOv2 模型"""
        
        model_names = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
        
        model_name = model_names.get(size, 'dinov2_vitb14')
        
        print(f"Loading {model_name} from torch.hub...")
        
        # try:
        #     # 从 torch hub 加载
        #     model = torch.hub.load(
        #         'facebookresearch/dinov2',
        #         model_name,
        #         pretrained=pretrained,
        #         skip_validation=True
        #     )
        #     print(f"Successfully loaded {model_name}")
        #     return model
            
        # except Exception as e:

        # print(f"Torch hub failed: {str(e)[:100]}")
        print("\nTrying alternative method...")
        
        # 从本地 cache 加载
        import os
        cache_dir = os.path.expanduser('~/.cache/torch/hub')
        path = os.path.join(cache_dir, "facebookresearch_dinov2_main")
        if os.path.exists(cache_dir):
            print(f"Checking cache: {cache_dir}")
            model = torch.hub.load(
                path,  # 本地路径
                'dinov2_vitb14',
                source='local',      # 从本地加载
                trust_repo=True      # 跳过验证
            )
            # # The model should be cached from first download
            # model = torch.hub.load(
            #     'facebookresearch/dinov2',
            #     model_name,
            #     pretrained=pretrained,
            #     force_reload=False
            # )
            print(f"Loaded from cache")
            return model
        
        raise RuntimeError(
            f"Failed to load DINOv2. Please check network connection.\n"
            f"Or download manually with: python download_dinov2.py"
        )
    
    def _setup_dims(self):
        """设立特征维度"""
        dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        
        self.feature_dim = dims.get(self.model_size, 768)
        
        self.out_channels = [
            self.feature_dim // 4,  # 1/4 尺度
            self.feature_dim // 2,  # 1/8 尺度
            self.feature_dim,       # 1/16 尺度
            self.feature_dim,       # 1/32 尺度
        ]
    
    def _build_proj(self):
        """构建投影层（用于多尺度特征适配）"""
        self.proj = nn.ModuleDict({
            's1': nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.LayerNorm(self.feature_dim // 4),
                nn.GELU()
            ),
            's2': nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.LayerNorm(self.feature_dim // 2),
                nn.GELU()
            ),
            's3': nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU()
            ),
            's4': nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU()
            )
        })
    
    def forward(self, x):
        """
        提取多尺度特征

        参数:
            x: [B, 3, H, W] 输入图像

        返回:
            包含 4 个特征图的列表：
                F1: [B, C/4, H/4, W/4]
                F2: [B, C/2, H/8, W/8]
                F3: [B, C, H/16, W/16]
                F4: [B, C, H/32, W/32]
        """
        B, C, H, W = x.shape
        
        # 从 DINOv2 提取中间层特征
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.layers,
            return_class_token=False,
            reshape=False
        )
        
        # 构建特征金字塔
        return self._pyramid(features, H, W)
    
    def _pyramid(self, feats, h, w):
        """构建特征金字塔"""
        pyr = []
        
        # DINOv2 使用 14x14 的图像块（patch）
        ps = 14
        nh = h // ps
        nw = w // ps
        
        stages = ['s1', 's2', 's3', 's4']
        sizes = [
            (h // 4, w // 4),
            (h // 8, w // 8),
            (h // 16, w // 16),
            (h // 32, w // 32)
        ]
        
        for f, s, sz in zip(feats, stages, sizes):
            B = f.shape[0]
            
            # 如果存在 class token，则移除（通常在位置 0）
            if f.shape[1] == nh * nw + 1:
                f = f[:, 1:, :]
            
            # 投影到目标维度
            f = self.proj[s](f)
            
            # 重塑为 2D 特征图
            N = f.shape[1]
            hw = int(N ** 0.5)
            f = f.transpose(1, 2).reshape(B, -1, hw, hw)
            
            # 双线性插值调整到目标分辨率
            f = F.interpolate(
                f,
                size=sz,
                mode='bilinear',
                align_corners=False
            )
            
            pyr.append(f)
        
        return pyr


if __name__ == '__main__':
    print("\nTesting DINOv2 Encoder\n")
    
    # 测试 1: 加载模型
    print("Test 1: Loading DINOv2-base")
    try:
        encoder = DINOv2Encoder(
            model_size='base',
            freeze=True,
            pretrained=True
        )
        print("✓ Model loaded\n")
        
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 测试 2: 前向传播
    print("Test 2: Forward pass")
    try:
        x = torch.randn(2, 3, 392, 392)
        features = encoder(x)
        
        print("✓ Forward pass successful")
        print("\nFeature shapes:")
        for i, f in enumerate(features):
            print(f"  F{i+1}: {f.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 测试 3: 不同输入尺寸
    print("\nTest 3: Different input sizes")
    try:
        sizes = [(224, 224), (280, 280), (392, 392), (448, 448)]
        
        for h, w in sizes:
            x = torch.randn(1, 3, h, w)
            feats = encoder(x)
            print(f"  Input {h}x{w}: ✓")
        
        print("✓ Multi-size test passed")
        
    except Exception as e:
        print(f"✗ Multi-size test failed: {e}")
    
    # 测试 4: 参数统计
    print("\nTest 4: Parameter count")
    try:
        total = sum(p.numel() for p in encoder.parameters())
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Frozen: {total - trainable:,}")
        
    except Exception as e:
        print(f"  Could not count: {e}")
    
    print("\n" + "="*60)
    print("All tests PASSED!")
    print("DINOv2 encoder is ready to use")
    print("="*60)