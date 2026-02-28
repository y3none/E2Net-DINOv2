import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttention(nn.Module):
    """
    内存高效的门控交叉注意力机制
    查询（Query）：来自精细细节特征（G'1）
    键（Key）与值（Value）：来自全局语义特征（G'4）
    """
    def __init__(self, dim, num_heads=8):
        super(GatedCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子，防止点积过大
        
        # 分别对 Query、Key、Value 进行线性投影
        self.q_proj = nn.Linear(dim, dim, bias=False)  # 来自细节特征
        self.k_proj = nn.Linear(dim, dim, bias=False)  # 来自语义特征
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # 门控机制：控制信息流动强度
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.norm_q = nn.LayerNorm(dim)  # Query 归一化
        self.norm_kv = nn.LayerNorm(dim)  # Key/Value 归一化
        
    def forward(self, query_feat, kv_feat):
        """
        参数:
            query_feat: 细节特征图 [B, C, H, W]
            kv_feat: 语义特征图 [B, C, H, W]
        返回:
            out: 融合后的特征图 [B, C, H, W]
        """
        B, C, H, W = query_feat.shape
        
        # 若空间尺寸过大，则下采样以节省显存
        max_size = 32  # 目前注意力计算的最大空间尺寸
        
        if H > max_size or W > max_size:
            query_down = F.adaptive_avg_pool2d(query_feat, (max_size, max_size))
            kv_down = F.adaptive_avg_pool2d(kv_feat, (max_size, max_size))
            H_down, W_down = max_size, max_size
        else:
            query_down = query_feat
            kv_down = kv_feat
            H_down, W_down = H, W
        
        # 展平空间维度：[B, C, H, W] -> [B, H*W, C]
        query = query_down.flatten(2).transpose(1, 2)
        kv = kv_down.flatten(2).transpose(1, 2)
        
        # 层归一化
        query = self.norm_q(query)
        kv = self.norm_kv(kv)
        
        # 投影并重塑为多头形式
        Q = self.q_proj(query).reshape(B, H_down*W_down, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, H_down*W_down, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, H_down*W_down, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算交叉注意力
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 对值加权求和
        out = (attn @ V).transpose(1, 2).reshape(B, H_down*W_down, C)
        
        # 应用门控机制
        gate_weights = self.gate(out)
        out = out * gate_weights
        
        out = self.proj(out)
        
        # 重塑回空间维度
        out = out.transpose(1, 2).reshape(B, C, H_down, W_down)
        
        # 若之前下采样，则上采样回原始尺寸
        if H_down != H or W_down != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # 与原始 Query 特征相加（残差连接）
        out = out + query_feat
        
        return out


class LightweightDecoder(nn.Module):
    """
    轻量级解码器：用于逐级上采样与特征细化
    """
    def __init__(self, in_channels=256, mid_channels=128):
        super(LightweightDecoder, self).__init__()
        
        # 第一级上采样（x2）
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 第二级上采样（x2），融合 G'3
        self.up2 = nn.Sequential(
            nn.Conv2d(mid_channels + in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 第三级细化（无上采样），融合 G'2
        self.up3 = nn.Sequential(
            nn.Conv2d(mid_channels + in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 最终预测头
        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, fused_feat, G_prime_2, G_prime_3):
        """
        参数:
            fused_feat: 门控交叉注意力融合后的特征 [B, C, H, W]
            G_prime_2: 中层特征 [B, C, H/2, W/2]
            G_prime_3: 高层特征 [B, C, H/4, W/4]
        返回:
            Y_final: 最终高分辨率分割图 [B, 1, H, W]
        """
        # 第一级上采样
        x = self.up1(fused_feat)  # [B, mid_ch, H*2, W*2]
        
        # 将 G'3 上采样至当前尺寸并拼接
        G_prime_3_resized = F.interpolate(G_prime_3, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, G_prime_3_resized], dim=1)
        x = self.up2(x)  # [B, mid_ch, H*4, W*4]
        
        #将 G'2 上采样至当前尺寸并拼接
        G_prime_2_resized = F.interpolate(G_prime_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, G_prime_2_resized], dim=1)
        x = self.up3(x)  # [B, mid_ch, H*4, W*4]
        
        # 生成最终预测
        Y_final = self.final_conv(x)
        
        return Y_final


class CCM(nn.Module):
    """
    认知确认模块（CCM，第三阶段）
    通过门控交叉注意力机制，融合局部细节与全局语义信息，
    精确确认伪装目标位置并生成最终分割结果。
    参数:
        channels: 统一通道维度（默认为 256）
        num_heads: 多头注意力的头数
    """
    def __init__(self, channels=256, num_heads=8):
        super(CCM, self).__init__()
        
        self.channels = channels
        
        # 门控交叉注意力：融合 G'1（细节）与 G'4（语义）
        self.gated_cross_attn = GatedCrossAttention(channels, num_heads)
        
        # 轻量级解码器：结合 G'2 和 G'3 进行多尺度细化
        self.decoder = LightweightDecoder(in_channels=channels, mid_channels=128)
    
    def forward(self, G_prime_features):
        """
        参数:
            G_prime_features: 经 CFZM 调制后的特征列表 [G'1, G'2, G'3, G'4]
        返回:
            Y_final: 最终高分辨率分割图 [B, 1, H, W]
        """
        G_prime_1, G_prime_2, G_prime_3, G_prime_4 = G_prime_features
        
        # 门控交叉注意力：Query来自G'1, Key&Value来自G'4
        fused_feat = self.gated_cross_attn(G_prime_1, G_prime_4)
        
        # 通过轻量解码器逐步上采样，并融合 G'2、G'3 实现多尺度细化
        Y_final = self.decoder(fused_feat, G_prime_2, G_prime_3)
        
        return Y_final


if __name__ == "__main__":
    # 测试 CCM 模块
    ccm = CCM(channels=256)
    
    # 模拟来自 CFZM 的调制特征
    G_prime_1 = torch.randn(2, 256, 56, 56)
    G_prime_2 = torch.randn(2, 256, 56, 56)
    G_prime_3 = torch.randn(2, 256, 56, 56)
    G_prime_4 = torch.randn(2, 256, 56, 56)
    
    G_prime_features = [G_prime_1, G_prime_2, G_prime_3, G_prime_4]
    
    Y_final = ccm(G_prime_features)
    
    print(f"Y_final shape: {Y_final.shape}")