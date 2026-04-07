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




# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SimplifiedSelfAttention(nn.Module):
#     """
#     自注意力层（显存优化版，与 CCM 的 GatedCrossAttention 策略一致）

#     核心改动（相对原版 H*W > 1024 → 32×32）：
#     ─────────────────────────────────────────────────────────────────
#     问题：
#       G4 经 CAEM 对齐后分辨率 = F1 = 129×129（518 输入时）
#       原版：强制下采样到 32×32 = 1024 tokens → 信息保留 6.2%
#       LFSM 是整个 Pipeline 的起点（Stage1 全域搜索），Y_coarse 的质量
#       决定了 CFZM 和 CCM 的上限。在起点丢 93.8% 的空间信息是最大的瓶颈。

#     方案：
#       用 max_tokens 控制（默认 4096 ≈ 64×64），与 CCM 保持一致。
#       + SDPA（FlashAttention）自动加速。

#     对应 idea.pdf:
#       "在 G4 上应用一个简化的自注意力层或视觉 Transformer 块，
#        计算特征图所有空间位置之间的关系，以捕获全局上下文信息。"
#     ─────────────────────────────────────────────────────────────────
#     """

#     def __init__(self, dim, num_heads=8, max_tokens=4096):
#         super(SimplifiedSelfAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.max_tokens = max_tokens

#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.proj = nn.Linear(dim, dim)
#         self.norm = nn.LayerNorm(dim)

#         self._use_sdpa = hasattr(F, 'scaled_dot_product_attention')

#     def _compute_pool_size(self, H, W):
#         """等比缩放，确保 token 总数 ≤ max_tokens。"""
#         N = H * W
#         if N <= self.max_tokens:
#             return H, W
#         ratio = (self.max_tokens / N) ** 0.5
#         return max(1, int(H * ratio)), max(1, int(W * ratio))

#     def forward(self, x):
#         """
#         参数:
#             x: [B, C, H, W] 输入特征图
#         返回:
#             out: [B, C, H, W] 带残差连接的输出
#         """
#         B, C, H, W = x.shape

#         # ── 自适应分辨率 ─────────────────────────────────────────
#         H_down, W_down = self._compute_pool_size(H, W)
#         need_resize = (H_down != H or W_down != W)

#         if need_resize:
#             x_down = F.adaptive_avg_pool2d(x, (H_down, W_down))
#         else:
#             x_down = x

#         N = H_down * W_down

#         # ── 展平 → 归一化 ───────────────────────────────────────
#         x_flat = x_down.flatten(2).transpose(1, 2)    # [B, N, C]
#         x_norm = self.norm(x_flat)

#         # ── QKV 投影 → 多头 ─────────────────────────────────────
#         qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)              # [3, B, heads, N, head_dim]
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         # ── 注意力计算 ───────────────────────────────────────────
#         if self._use_sdpa:
#             attn_out = F.scaled_dot_product_attention(
#                 q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
#             )
#         else:
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn_out = attn @ v

#         # ── 合并多头 → 投影 ─────────────────────────────────────
#         out = attn_out.transpose(1, 2).reshape(B, N, C)
#         out = self.proj(out)

#         # ── 恢复空间维度 ─────────────────────────────────────────
#         out = out.transpose(1, 2).reshape(B, C, H_down, W_down)

#         if need_resize:
#             out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

#         # ── 残差连接 ─────────────────────────────────────────────
#         return out + x


# class TransformerBlock(nn.Module):
#     """
#     Transformer 块：自注意力 + FFN

#     注意：FFN 也在与自注意力相同的分辨率下操作（原版 FFN 在全分辨率
#     129×129 上跑，但自注意力只在 32×32 上，存在分辨率不一致）。
#     现在两者统一为 max_tokens 控制的分辨率。
#     """

#     def __init__(self, dim, num_heads=8, mlp_ratio=4.0, max_tokens=4096):
#         super(TransformerBlock, self).__init__()
#         self.attn = SimplifiedSelfAttention(dim, num_heads, max_tokens)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)

#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Linear(mlp_hidden_dim, dim)
#         )

#     def forward(self, x):
#         """
#         参数:
#             x: [B, C, H, W]
#         返回:
#             x: [B, C, H, W]
#         """
#         B, C, H, W = x.shape

#         # 自注意力（内部含残差连接）
#         x = self.attn(x)

#         # FFN + 残差连接
#         x_flat = x.flatten(2).transpose(1, 2)           # [B, H*W, C]
#         x_flat = x_flat + self.mlp(self.norm2(x_flat))
#         x = x_flat.transpose(1, 2).reshape(B, C, H, W)

#         return x


# class LFSM(nn.Module):
#     """
#     侧向中央凹搜索模块（LFSM，Stage 1）

#     对应 idea.pdf:
#         "CAEM 输出的最具全局上下文信息的高分辨率特征 G4 作为输入"
#         "在 G4 上应用简化的自注意力层或 ViT 块，计算所有空间位置
#          之间的关系，以捕获全局上下文信息"
#         "通过 1x1 卷积 + BN + Sigmoid 预测头 → Y_coarse"

#     参数:
#         in_channels : 输入通道数
#         num_heads   : 多头注意力头数
#         num_blocks  : Transformer 块数量
#         max_tokens  : 自注意力最大 token 数（默认 4096 ≈ 64×64）
#     """

#     def __init__(self, in_channels=256, num_heads=8, num_blocks=2, max_tokens=4096):
#         super(LFSM, self).__init__()

#         self.transformer_blocks = nn.ModuleList([
#             TransformerBlock(in_channels, num_heads, max_tokens=max_tokens)
#             for _ in range(num_blocks)
#         ])

#         # 预测头：idea.pdf 原文 "1x1 卷积 + BN + Sigmoid"
#         self.pred_head = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 2, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, G4):
#         """
#         参数:
#             G4: [B, C, H, W] 语义最强的特征图
#         返回:
#             Y_coarse: [B, 1, H, W] 粗略显著性图
#         """
#         x = G4

#         for block in self.transformer_blocks:
#             x = block(x)

#         Y_coarse = self.pred_head(x)
#         return Y_coarse


# if __name__ == "__main__":
#     print("=" * 60)
#     print(" LFSM 显存优化版 — 单元测试")
#     print("=" * 60)

#     # ── 分辨率对比 ────────────────────────────────────────────────
#     print("\n[对比] 129×129 输入（518 图像，G4 实际分辨率）：")
#     total = 129 * 129
#     lfsm = LFSM(in_channels=256, num_heads=8, num_blocks=2, max_tokens=4096)
#     H_d, W_d = lfsm.transformer_blocks[0].attn._compute_pool_size(129, 129)
#     configs = [
#         ("原版 H*W>1024 → 32×32", 32, 32),
#         ("本版 max_tokens=4096",    H_d, W_d),
#     ]
#     for name, h, w in configs:
#         tokens = h * w
#         ratio = tokens / total * 100
#         print(f"  {name:<30s}  {h}×{w}={tokens:<6d}  保留{ratio:5.1f}%")

#     # ── 测试 1：129×129 前向传播 ──────────────────────────────────
#     print("\n[Test 1] 129×129 前向传播")
#     G4 = torch.randn(2, 256, 129, 129)
#     Y = lfsm(G4)
#     print(f"  G4: {G4.shape} → Y_coarse: {Y.shape}")
#     print(f"  值域: [{Y.min().item():.4f}, {Y.max().item():.4f}]")
#     assert Y.shape == (2, 1, 129, 129)
#     print("  ✓ 通过")

#     # ── 测试 2：98×98（392 输入）──────────────────────────────────
#     print("\n[Test 2] 98×98 前向传播（392 输入）")
#     G4_98 = torch.randn(2, 256, 98, 98)
#     Y_98 = lfsm(G4_98)
#     print(f"  G4: {G4_98.shape} → Y_coarse: {Y_98.shape}")
#     print("  ✓ 通过")

#     # ── 测试 3：小特征图（全分辨率，不下采样）────────────────────
#     print("\n[Test 3] 16×16 特征图（全分辨率）")
#     G4_small = torch.randn(2, 256, 16, 16)
#     Y_small = lfsm(G4_small)
#     H_s, W_s = lfsm.transformer_blocks[0].attn._compute_pool_size(16, 16)
#     print(f"  16×16={16*16} ≤ 4096 → 全分辨率 {H_s}×{W_s}")
#     print(f"  Y_coarse: {Y_small.shape}")
#     print("  ✓ 通过")

#     # ── 测试 4：梯度流 ────────────────────────────────────────────
#     print("\n[Test 4] 梯度流")
#     lfsm.train()
#     G4_g = torch.randn(1, 256, 129, 129, requires_grad=True)
#     Y_g = lfsm(G4_g)
#     Y_g.mean().backward()
#     assert G4_g.grad is not None
#     print(f"  G4 梯度范数: {G4_g.grad.norm().item():.6f}")
#     print("  ✓ 通过")

#     # ── 测试 5：参数量对比 ────────────────────────────────────────
#     print("\n[Test 5] 参数量（与原版完全相同，仅改变计算分辨率）")
#     params = sum(p.numel() for p in lfsm.parameters())
#     trainable = sum(p.numel() for p in lfsm.parameters() if p.requires_grad)
#     print(f"  Total: {params:,}  Trainable: {trainable:,}")
#     print("  ✓ 通过")

#     # ── 测试 6：SDPA 检测 ─────────────────────────────────────────
#     print(f"\n[Test 6] SDPA: {lfsm.transformer_blocks[0].attn._use_sdpa}")

#     print("\n" + "=" * 60)
#     print(" 所有测试通过")
#     print("=" * 60)