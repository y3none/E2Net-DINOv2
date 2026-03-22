import torch
import torch.nn as nn
import torch.nn.functional as F

class CFZM(nn.Module):
    """
    中央中央凹缩放模块（CFZM，第二阶段）
    利用粗略注意力图动态聚焦，自适应地增强所有尺度的特征，
    重点强化可疑区域的表征能力。
    参数:
        channels: 统一的通道维度（默认为 256）
    """
    def __init__(self, channels=256):
        super(CFZM, self).__init__()
        
        self.channels = channels
        
        # 注意力调制后的特征增强卷积
        self.enhance_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # 每一级的细化显著性图预测头
        self.pred_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])
    
    def forward(self, G_features, Y_coarse):
        """
        参数:
            G_features: 来自 CAEM 的对齐特征列表 [G1, G2, G3, G4]
            Y_coarse: 来自 LFSM 的粗略显著性图 [B, 1, H, W]
        返回:
            G_prime_features: 经注意力调制后的特征列表 [G'1, G'2, G'3, G'4]
            Y_refined: 细化后的显著性图（来自最高分辨率层级）[B, 1, H, W]
            Y_maps: 所有层级的辅助预测图列表（用于多尺度监督）
        """
        G1, G2, G3, G4 = G_features
        
        G_prime_features = []
        Y_maps = []
        
        # 遍历每一级特征进行动态聚焦
        for i, Gi in enumerate(G_features):
            # 步骤1: 上采样 Y_coarse 得到 Ai
            # Ai: [B, 1, H, W] - 单通道空间注意力图
            Ai = F.interpolate(Y_coarse, size=Gi.shape[2:], mode='bilinear', align_corners=False)
            
            # 步骤2: 应用 Sigmoid
            # 注意：Y_coarse 可能已经经过 Sigmoid，这里再次应用是为了确保值在 [0,1]
            # 如果 Y_coarse 已经是 [0,1]，可以跳过这步或使用其他激活
            attention_weight = torch.sigmoid(Ai)  # [B, 1, H, W]
            
            # 步骤3: 论文公式 G'i = Gi ⊗ σ(Ai) + Gi
            # attention_weight [B, 1, H, W] 会自动广播到 [B, C, H, W]
            Gi_prime = Gi * attention_weight + Gi  # 逐元素乘法 + 残差
            
            # 步骤4: 特征增强
            Gi_prime = self.enhance_convs[i](Gi_prime)
            
            G_prime_features.append(Gi_prime)
            
            # 辅助预测
            Yi = self.pred_heads[i](Gi_prime)
            Y_maps.append(Yi)
        
        # 最终细化预测图取自最高分辨率层级（G'1）
        Y_refined = Y_maps[0]
        
        return G_prime_features, Y_refined, Y_maps


if __name__ == "__main__":
    # 测试 CFZM 模块
    cfzm = CFZM(channels=256)
    
    # 模拟来自 CAEM 的对齐特征
    G1 = torch.randn(2, 256, 56, 56)
    G2 = torch.randn(2, 256, 56, 56)
    G3 = torch.randn(2, 256, 56, 56)
    G4 = torch.randn(2, 256, 56, 56)
    G_features = [G1, G2, G3, G4]
    
    # 模拟来自 LFSM 的粗略显著性图
    Y_coarse = torch.randn(2, 1, 56, 56)
    
    G_prime_features, Y_refined, Y_maps = cfzm(G_features, Y_coarse)
    
    print("G' features:")
    for i, G_prime in enumerate(G_prime_features):
        print(f"G'{i+1} shape: {G_prime.shape}")
    
    print(f"\nY_refined shape: {Y_refined.shape}")
    print(f"Number of auxiliary predictions: {len(Y_maps)}")
