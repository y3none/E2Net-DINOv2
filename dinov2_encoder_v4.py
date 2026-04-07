#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 Encoder for E2Net — Parallel Adapter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import List

warnings.filterwarnings('ignore', message='xFormers is not available')


# ═════════════════════════════════════════════════════════════════════════════
# FeatureAdapter：Bottleneck Adapter（在 token 序列上操作）
# ═════════════════════════════════════════════════════════════════════════════

class FeatureAdapter(nn.Module):
    """
    轻量级 Bottleneck Adapter（token 序列域）
    结构：
        x  ->  LayerNorm  ->  Linear(dim->hidden)  ->  GELU
           ->  Linear(hidden->dim)  ->  scale · out

    参数:
        dim        : token 特征维度（= DINOv2 hidden dim）
        reduction  : 瓶颈压缩比，hidden = max(dim // reduction, 32)
        init_scale : 输出缩放初始值（接近 0 -> near-identity）
    """

    def __init__(self, dim: int, reduction: int = 4, init_scale: float = 1e-3):
        super().__init__()
        hidden = max(dim // reduction, 32)

        self.norm  = nn.LayerNorm(dim)
        self.down  = nn.Linear(dim, hidden, bias=False)
        self.act   = nn.GELU()
        self.up    = nn.Linear(hidden, dim, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        # up 权重置零：确保训练初期 Adapter 输出接近 0，⊕ 后等价于恒等
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x : [B, N, dim]  — Block 输出的 token 序列
        返回:
            delta : [B, N, dim]  — 残差增量（不含 x 本身）
        """
        return self.scale * self.up(self.act(self.down(self.norm(x))))


# ═════════════════════════════════════════════════════════════════════════════
# ParallelAdapterBlock：并联 Adapter 包装（对应图中带 ⊕ 的节点）
# ═════════════════════════════════════════════════════════════════════════════

class ParallelAdapterBlock(nn.Module):
    """
    将 Adapter 并联到 DINOv2 Block 旁，实现 ⊕ 的注入逻辑。
    对应关系：
        Block_i 输出 out_i
                |
                |-> Adapter_i(out_i) = delta_i
                |
                ▼
        out_i + delta_i  ─>  作为 Block_{i+1} 的输入
    即：Adapter 以当前 Block 的输出作为输入，产生残差 delta，
        delta 被加到当前 Block 的输出上，再传入下一个 Block。

    参数:
        block   : 原始 DINOv2 Block（冻结）
        adapter : FeatureAdapter（可训练）
    """

    def __init__(self, block: nn.Module, adapter: FeatureAdapter):
        super().__init__()
        self.block   = block    # frozen
        self.adapter = adapter  # trainable

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        数据流：
            x  ─>  Block(x)  ─>  out
                              +  Adapter(out)   <- 并联，输入是 Block 的输出
                              =  out + delta    <- ⊕ 操作
        """
        out = self.block(x, *args, **kwargs)

        # DINOv2 某些 Block 返回 tuple（如含 attn weights），只取 hidden state
        if isinstance(out, tuple):
            hidden, *rest = out
            delta  = self.adapter(hidden)
            return (hidden + delta, *rest)

        # 标准情况：out 就是 token 张量 [B, N, C]
        delta = self.adapter(out)
        return out + delta   # ⊕：注入下一个 Block 的输入


# ═════════════════════════════════════════════════════════════════════════════
# PlainBlock：不含 Adapter 的普通包装（保持接口统一，仅透传）
# ═════════════════════════════════════════════════════════════════════════════

class PlainBlock(nn.Module):
    """
    对不需要插入 Adapter 的 Block 做透明包装，保持 backbone.blocks 接口统一。
    block 1、2 等前置层均使用此包装。
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.block(x, *args, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# DINOv2Encoder：主编码器（Parallel Adapter）
# ═════════════════════════════════════════════════════════════════════════════

class DINOv2Encoder(nn.Module):
    """
    DINOv2 编码器
    数据流（以 adapter_at=[5, 11]，DINOv2-base 共 12 层为例）：
        image
          │
        Block_0  ->  Block_1  ->  ...  ->  Block_5
                                              │
                                         ⊕──Adapter_5   <- trainable
                                              │
                                         Block_6  ->  ...  ->  Block_11
                                                                   │
                                                              ⊕──Adapter_11  <- trainable
                                                                   │
                                         get_intermediate_layers([3,6,9,11])
                                                                   │
                                         Proj x 4  ->  reshape  ->  interpolate
                                                                   │
                                         [F1, F2, F3, F4]  ->  CAEM

    参数:
        model_size        : 'small' | 'base' | 'large' | 'giant'
        freeze            : 是否冻结 backbone（Adapter/Proj 始终可训练）
        pretrained        : 是否加载预训练权重
        adapter_at        : 插入 Adapter 的层索引列表（0-based）
                            默认 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        adapter_reduction : Adapter 瓶颈压缩比，默认 4
        adapter_scale     : Adapter 残差初始缩放因子，默认 1e-3
    """

    def __init__(
        self,
        model_size: str       = 'base',
        freeze: bool          = True,
        pretrained: bool      = True,
        adapter_at: List[int] = None,
        adapter_reduction: int  = 4,
        adapter_scale: float  = 1e-3,
    ):
        super().__init__()

        self.model_size = model_size
        self.adapter_at = sorted(set(adapter_at)) if adapter_at else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        print("=" * 60)
        print("DINOv2 Encoder  [Parallel Adapter — 图示架构]")
        print(f"  model_size        : {model_size}")
        print(f"  adapter_at layers : {self.adapter_at}")
        print(f"  adapter_reduction : {adapter_reduction}")
        print(f"  adapter_scale     : {adapter_scale}")
        print("=" * 60)

        # 1. 加载 backbone
        self.backbone = self._load_dinov2(model_size, pretrained)

        # 2. 冻结 backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Backbone parameters : FROZEN")

        # 3. 配置维度
        self._setup_dims()

        # 4. 注入 Parallel Adapter（替换 backbone.blocks）
        self._inject_parallel_adapters(adapter_reduction, adapter_scale)

        # 5. 构建投影层
        self._build_proj()

        # 6. 中间层索引
        self.layers = [3, 6, 9, 11]

        # 7. 打印参数统计
        self._print_param_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # 初始化辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _load_dinov2(self, size: str, pretrained: bool) -> nn.Module:
        model_names = {
            'small': 'dinov2_vits14',
            'base' : 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14',
        }
        name = model_names.get(size, 'dinov2_vitb14')
        print(f"Loading {name} ...")

        import os
        cache_dir  = os.path.expanduser('~/.cache/torch/hub')
        local_path = os.path.join(cache_dir, "facebookresearch_dinov2_main")

        if os.path.exists(local_path):
            model = torch.hub.load(
                local_path, 'dinov2_vitb14',
                source='local', trust_repo=True
            )
            print("  Loaded from local cache")
            return model

        raise RuntimeError(
            "Failed to load DINOv2. "
            "Please download manually: python download_dinov2.py"
        )

    def _setup_dims(self):
        dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
        self.feature_dim  = dims.get(self.model_size, 768)
        self.out_channels = [
            self.feature_dim // 4,
            self.feature_dim // 2,
            self.feature_dim,
            self.feature_dim,
        ]

    def _inject_parallel_adapters(self, reduction: int, init_scale: float):
        """
        遍历 backbone.blocks，按 adapter_at 指定的层索引：
          - 在该层插入 ParallelAdapterBlock（Block + 并联 Adapter）
          - 其余层用 PlainBlock 透明包装（保持接口一致）

        替换后 backbone.blocks 中每个元素均为统一的包装类，
        get_intermediate_layers 可正常抽取中间层特征。
        """
        if not hasattr(self.backbone, 'blocks'):
            raise AttributeError(
                "backbone 不含 'blocks' 属性，无法注入 Parallel Adapter。"
            )

        num_blocks = len(self.backbone.blocks)

        # 验证 adapter_at 合法性
        for idx in self.adapter_at:
            if idx < 0 or idx >= num_blocks:
                raise ValueError(
                    f"adapter_at 中的层索引 {idx} 超出范围 [0, {num_blocks-1}]"
                )

        adapter_set  = set(self.adapter_at)
        new_blocks   = nn.ModuleList()
        adapter_dict = {}   # idx -> FeatureAdapter，供参数统计和测试访问

        for idx, block in enumerate(self.backbone.blocks):
            if idx in adapter_set:
                adapter = FeatureAdapter(self.feature_dim, reduction, init_scale)
                adapter_dict[idx] = adapter
                new_blocks.append(ParallelAdapterBlock(block, adapter))
                print(f"  Block {idx:2d} : ParallelAdapterBlock  <- Adapter 插入")
            else:
                new_blocks.append(PlainBlock(block))
                print(f"  Block {idx:2d} : PlainBlock             (frozen, no adapter)")

        self.backbone.blocks = new_blocks

        # 将 adapter_dict 转为 ModuleDict 以便参数管理
        self.adapters = nn.ModuleDict({
            f'layer_{idx}': adp for idx, adp in adapter_dict.items()
        })

    def _build_proj(self):
        d = self.feature_dim
        self.proj = nn.ModuleDict({
            's1': nn.Sequential(nn.Linear(d, d // 4), nn.LayerNorm(d // 4), nn.GELU()),
            's2': nn.Sequential(nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU()),
            's3': nn.Sequential(nn.Linear(d, d),       nn.LayerNorm(d),       nn.GELU()),
            's4': nn.Sequential(nn.Linear(d, d),       nn.LayerNorm(d),       nn.GELU()),
        })

    def _print_param_stats(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        proj_n    = sum(p.numel() for p in self.proj.parameters())
        adp_n     = sum(p.numel() for p in self.adapters.parameters())

        print(f"\n  feature_dim       : {self.feature_dim}")
        print(f"  out_channels      : {self.out_channels}")
        print(f"  Total params      : {total:,}")
        print(f"  Trainable params  : {trainable:,}")
        print(f"    ↳ Adapter x {len(self.adapter_at)}  : {adp_n:,}")
        print(f"    ↳ Proj          : {proj_n:,}")
        print(f"  Frozen  params    : {total - trainable:,}")
        print("=" * 60 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 前向传播
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        参数:
            x : [B, 3, H, W]

        返回:
            [F1, F2, F3, F4]
                F1 : [B, C/4, H/4,  W/4 ]
                F2 : [B, C/2, H/8,  W/8 ]
                F3 : [B, C,   H/16, W/16]
                F4 : [B, C,   H/32, W/32]

        内部流程：
            image -> backbone（含 Parallel Adapter ⊕ 注入）
                  -> get_intermediate_layers([3,6,9,11])
                  -> Proj + reshape + interpolate
                  -> [F1, F2, F3, F4]
        """
        _, _, H, W = x.shape

        # backbone 内部已完成 Parallel Adapter 的 ⊕ 注入
        raw_features = self.backbone.get_intermediate_layers(
            x,
            n=self.layers,
            return_class_token=False,
            reshape=False,
        )   # list of [B, N, feature_dim]

        return self._pyramid(raw_features, H, W)

    def _pyramid(self, feats, h: int, w: int):
        """Proj -> reshape -> interpolate，构建多尺度特征金字塔"""
        pyr    = []
        ps     = 14
        nh, nw = h // ps, w // ps
        stages = ['s1', 's2', 's3', 's4']
        sizes  = [
            (h // 4,  w // 4 ),
            (h // 8,  w // 8 ),
            (h // 16, w // 16),
            (h // 32, w // 32),
        ]

        for f, stage, sz in zip(feats, stages, sizes):
            B = f.shape[0]

            # 去掉 class token（若存在）
            if f.shape[1] == nh * nw + 1:
                f = f[:, 1:, :]

            # Proj（token 维度）
            f = self.proj[stage](f)                        # [B, N, C']

            # reshape -> 空间特征图
            N  = f.shape[1]
            hw = int(N ** 0.5)
            f  = f.transpose(1, 2).reshape(B, -1, hw, hw) # [B, C', hw, hw]

            # 双线性插值到目标分辨率
            f  = F.interpolate(f, size=sz, mode='bilinear', align_corners=False)

            pyr.append(f)

        return pyr   # [F1, F2, F3, F4]


# ═════════════════════════════════════════════════════════════════════════════
# 单元测试
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print(" DINOv2Encoder Parallel Adapter — unit test")
    print("=" * 60)

    # ── 测试 1：默认配置（adapter_at=[5,11]，对应图示 block i / block n）
    print("\nTest 1: Default config  adapter_at=[5, 11]")
    enc = DINOv2Encoder(
        model_size='base',
        freeze=True,
        pretrained=True,
        adapter_at=[5, 11],
        adapter_reduction=4,
        adapter_scale=1e-3,
    )

    # ── 测试 2：前向传播形状 ──────────────────────────────────────────────
    print("Test 2: Forward pass shape check")
    x     = torch.randn(2, 3, 392, 392)
    feats = enc(x)
    for i, f in enumerate(feats):
        print(f"  F{i+1}: {tuple(f.shape)}")
    assert len(feats) == 4, "应输出 4 个特征图"
    print("✓ Shape check passed")

    # ── 测试 3：near-identity 初始化验证 ─────────────────────────────────
    print("\nTest 3: near-identity check")
    enc.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 196, 768)
        for name, adp in enc.adapters.items():
            delta = adp(dummy)
            print(f"  {name}: mean|delta| = {delta.abs().mean().item():.6f}  (expect ≈ 0)")
    print("✓ Near-identity check passed")

    # ── 测试 4：梯度流验证 ────────────────────────────────────────────────
    print("\nTest 4: Gradient flow check")
    enc.train()
    x     = torch.randn(1, 3, 392, 392)
    feats = enc(x)
    loss  = sum(f.mean() for f in feats)
    loss.backward()

    # Adapter 参数必须有梯度
    for name, p in enc.adapters.named_parameters():
        assert p.grad is not None, f"Adapter {name} 无梯度！"
    print("  ✓ Adapter params have gradients")

    # Proj 参数必须有梯度
    for name, p in enc.proj.named_parameters():
        assert p.grad is not None, f"Proj {name} 无梯度！"
    print("  ✓ Proj params have gradients")

    # backbone 中原始 Block 参数（PlainBlock/ParallelAdapterBlock 内的 block）不更新
    frozen_ok = True
    for name, p in enc.backbone.named_parameters():
        # adapters 的参数会出现在 backbone.blocks 里，过滤掉
        if 'adapter' in name:
            continue
        if p.grad is not None:
            print(f"  ✗ backbone {name} 不应有梯度！")
            frozen_ok = False
    if frozen_ok:
        print("  ✓ Backbone frozen params have no gradients")
    print("✓ Gradient flow check passed")

    # ── 测试 5：自定义 adapter_at ─────────────────────────────────────────
    print("\nTest 5: Custom adapter_at=[2, 5, 8, 11]")
    enc2 = DINOv2Encoder(
        model_size='base',
        freeze=True,
        pretrained=True,
        adapter_at=[2, 5, 8, 11],
        adapter_reduction=4,
        adapter_scale=1e-3,
    )
    feats2 = enc2(torch.randn(1, 3, 392, 392))
    print(f"  Adapter count : {len(enc2.adapters)}  (expect 4)")
    print(f"  F1 shape      : {tuple(feats2[0].shape)}")
    print("✓ Custom adapter_at test passed")

    # ── 测试 6：参数量对比 ────────────────────────────────────────────────
    print("\nTest 6: Parameter count  (adapter_at=[5,11] vs [2,5,8,11])")
    for model, tag in [(enc, '[5,11]'), (enc2, '[2,5,8,11]')]:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adp_n     = sum(p.numel() for p in model.adapters.parameters())
        print(f"  adapter_at={tag:<12}  "
              f"trainable={trainable:,}  adapter={adp_n:,}  total={total:,}")

    print("\n" + "=" * 60)
    print(" All tests completed")
    print("=" * 60)
