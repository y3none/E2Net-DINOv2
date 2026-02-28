import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    轻量级通道注意力机制（如 SENet 或 ECANet）
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeatureFusion(nn.Module):
    """
    特征融合模块：用于融合深层语义特征与浅层高分辨率特征
    """
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_attn = ChannelAttention(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.channel_attn(x)
        return x


class CAEM(nn.Module):
    """
    跨尺度特征对齐与增强模块（CAEM）

    将 DINOv2 提取的多尺度特征对齐到同一分辨率，并通过将深层语义信息注入浅层高分辨率特征，
    实现多层次特征的增强。

    参数:
        in_channels: 输入特征通道数列表，对应 [F1, F2, F3, F4]
        unified_channels: 统一后的输出通道维度
    """
    def __init__(self, in_channels=[192, 384, 768, 768], unified_channels=256):
        super(CAEM, self).__init__()
        
        self.in_channels = in_channels
        self.unified_channels = unified_channels
        
        # 为每一级特征定义通道压缩卷积
        self.reduce_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, unified_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(unified_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels
        ])
        
        # 逐级融合模块
        # 将 F4 与 F3 融合
        self.fuse_4_3 = FeatureFusion(unified_channels * 2, unified_channels)
        # 将 (F4+F3) 与 F2 融合
        self.fuse_3_2 = FeatureFusion(unified_channels * 2, unified_channels)
        # 将 ((F4+F3)+F2) 与 F1 融合
        self.fuse_2_1 = FeatureFusion(unified_channels * 2, unified_channels)
        
        # 每个对齐后特征的输出卷积
        self.out_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(unified_channels, unified_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(unified_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
    
    def forward(self, features):
        """
        参数:
            features: 来自 DINOv2 编码器的多尺度特征列表 [F1, F2, F3, F4]
        返回:
        """
        F1, F2, F3, F4 = features
        
        # 目标分辨率：采用最高分辨率（F1 的尺寸）
        target_size = F1.shape[2:]
        
        # 步骤 1：对所有特征进行通道压缩
        F1_reduced = self.reduce_convs[0](F1)
        F2_reduced = self.reduce_convs[1](F2)
        F3_reduced = self.reduce_convs[2](F3)
        F4_reduced = self.reduce_convs[3](F4)
        
        # 步骤 2：自底向上逐级上采样并融合
        # 从最深层特征 F4 开始
        
        # 将 F4 上采样至 F3 尺寸，并与 F3 融合
        F4_up = F.interpolate(F4_reduced, size=F3_reduced.shape[2:], mode='bilinear', align_corners=False)
        F4_3 = self.fuse_4_3(torch.cat([F4_up, F3_reduced], dim=1))
        
        # 将融合结果上采样至 F2 尺寸，并与 F2 融合
        F4_3_up = F.interpolate(F4_3, size=F2_reduced.shape[2:], mode='bilinear', align_corners=False)
        F4_3_2 = self.fuse_3_2(torch.cat([F4_3_up, F2_reduced], dim=1))
        
        # 将融合结果上采样至 F1 尺寸，并与 F1 融合
        F4_3_2_up = F.interpolate(F4_3_2, size=F1_reduced.shape[2:], mode='bilinear', align_corners=False)
        F4_3_2_1 = self.fuse_2_1(torch.cat([F4_3_2_up, F1_reduced], dim=1))
        
        # 步骤 3：生成统一分辨率下的多级对齐特征
        
        # G1：包含最丰富的细节 + 全尺度语义信息
        G1 = self.out_convs[0](F4_3_2_1)
        
        # G2：中层特征 + 深层语义信息
        F4_3_2_for_G2 = F.interpolate(F4_3_2, size=target_size, mode='bilinear', align_corners=False)
        G2 = self.out_convs[1](F4_3_2_for_G2)
        
        # G3：高层特征 + 深层语义信息
        F4_3_for_G3 = F.interpolate(F4_3, size=target_size, mode='bilinear', align_corners=False)
        G3 = self.out_convs[2](F4_3_for_G3)
        
        # G4：最强语义特征（来自最深层）
        F4_for_G4 = F.interpolate(F4_reduced, size=target_size, mode='bilinear', align_corners=False)
        G4 = self.out_convs[3](F4_for_G4)
        
        return [G1, G2, G3, G4]


if __name__ == "__main__":
    # 测试 CAEM 模块
    caem = CAEM(in_channels=[192, 384, 768, 768], unified_channels=256)
    
    # 模拟 DINOv2 输出的多尺度特征
    F1 = torch.randn(2, 192, 56, 56)  # 最高分辨率（浅层）
    F2 = torch.randn(2, 384, 28, 28)
    F3 = torch.randn(2, 768, 14, 14)
    F4 = torch.randn(2, 768, 7, 7)    # 最低分辨率（深层，语义最强）
    
    features = [F1, F2, F3, F4]
    aligned_features = caem(features)
    
    print("Aligned features:")
    for i, G in enumerate(aligned_features):
        print(f"G{i+1} shape: {G.shape}")
