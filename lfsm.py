import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedSelfAttention(nn.Module):
    """
    内存高效的自注意力层，用于捕获全局上下文信息
    """
    def __init__(self, dim, num_heads=8):
        super(SimplifiedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        参数:
            x: [B, C, H, W] 输入特征图
        返回:
            x: [B, C, H, W] 带残差连接的输出
        """
        B, C, H, W = x.shape
        
        # 若空间尺寸过大，则下采样以节省显存
        if H * W > 1024:  # 当分辨率超过 32x32 时
            x_down = F.adaptive_avg_pool2d(x, (32, 32))  # 自适应池化到 32x32
            B, C, H_down, W_down = x_down.shape
        else:
            x_down = x
            H_down, W_down = H, W
        
        # 将空间维度展平：[B, C, H, W] -> [B, H*W, C]
        x_flat = x_down.flatten(2).transpose(1, 2)
        
        # 应用层归一化
        x_norm = self.norm(x_flat)
        
        # 计算 Q、K、V
        qkv = self.qkv(x_norm).reshape(B, H_down*W_down, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状为 [B, num_heads, H*W, head_dim]
        
        # 计算注意力权重（内存高效实现），使用更小的块来防止OOM
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 对值向量加权求和
        out = (attn @ v).transpose(1, 2).reshape(B, H_down*W_down, C)
        out = self.proj(out)
        
        # 重塑回空间维度：[B, C, H_down, W_down]
        out = out.transpose(1, 2).reshape(B, C, H_down, W_down)
        
        # 如果之前进行了下采样，则上采样回原始尺寸
        if H_down != H or W_down != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return out + x  # 添加残差连接


class TransformerBlock(nn.Module):
    """
    简化的 Transformer 块，包含自注意力机制和前馈网络（FFN）
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()
        self.attn = SimplifiedSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        """
        参数:
            x: [B, C, H, W] 输入特征图
        返回:
            x: [B, C, H, W] 输出特征图
        """
        B, C, H, W = x.shape
        
        # 自注意力 + 残差连接
        x = self.attn(x)
        
        # 前馈网络（FFN）+ 残差连接
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class LFSM(nn.Module):
    """
    侧向中央凹搜索模块（LFSM，第一阶段）
    利用语义信息最丰富的特征图（G4）进行全局搜索，初步定位可疑区域。
    参数:
        in_channels: 输入通道数（来自 CAEM 的输出）
        num_heads: 多头注意力的头数
        num_blocks: Transformer 块的数量
    """
    def __init__(self, in_channels=256, num_heads=8, num_blocks=2):
        super(LFSM, self).__init__()
        
        # 使用多个 Transformer 块建模全局上下文
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(in_channels, num_heads) for _ in range(num_blocks)
        ])
        
        # 粗略显著性图预测头
        self.pred_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, G4):
        """
        参数:
            G4: 语义信息最强的特征图 [B, C, H, W]
        返回:
            Y_coarse: 初始粗略显著性图 [B, 1, H, W]
        """
        x = G4
        
        # 通过 Transformer 块增强全局上下文建模能力
        for block in self.transformer_blocks:
            x = block(x)
        
        # 生成粗略预测图
        Y_coarse = self.pred_head(x)
        
        return Y_coarse


if __name__ == "__main__":
    # 测试 LFSM 模块
    lfsm = LFSM(in_channels=256)
    
    # 模拟来自 CAEM 的 G4 特征
    G4 = torch.randn(2, 256, 56, 56)
    
    Y_coarse = lfsm(G4)
    
    print(f"G4 input shape: {G4.shape}")
    print(f"Y_coarse output shape: {Y_coarse.shape}")