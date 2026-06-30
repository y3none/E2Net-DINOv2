#!/bin/bash
# 训练 E2Net with DINOv2

# 设置环境变量来阻止warnings
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "============================================================"
echo "Training E2Net with DINOv2 Encoder"
echo "============================================================"

# 训练配置
DATAPATH="../dataset/TrainDataset"
VAL_DATAPATH="../dataset/TestDataset/CAMO"
BATCH_SIZE=4
EPOCHS=100
LR=0.0001
IMAGE_SIZE=392  # DINOv2 必须使用14的倍数

# 模型配置
ENCODER_SIZE="base"  # 选项: small、base、large、giant
UNIFIED_CHANNELS=256

# adapter_at: 在哪几层插入 Parallel Adapter（0-based，对应图示 block i/n）
ADAPTER_AT="0 1 2 3 4 5 6 7 8 9 10 11"
ADAPTER_REDUCTION=4
# adapter_scale: 残差初始缩放因子
#   小数据集（CHAMELEON ~76张）→ 1e-4（保守，防过拟合）
#   大数据集（COD10K）        → 1e-2（积极，加快适配）
#   混合训练（默认）           → 1e-3（折中）
ADAPTER_SCALE=1e-3

# 损失权重
LAMBDA_DICE=1.0
LAMBDA_BCE=1.0
LAMBDA_COARSE=0.5
LAMBDA_REFINED=0.3

# lambda_edge=0 可完全关闭，等价于原版损失
# 推荐范围 0.2~0.5；CHAMELEON 细腻纹理场景建议 0.3~0.5
LAMBDA_EDGE=0.3
EDGE_DILATION=3       # 边界区域膨胀像素数，覆盖模糊边界不确定区域

# Checkpoint
CHECKPOINT_DIR="checkpoint/E2Net_DINOv2-v1_7"
SAVE_FREQ=10

# Device
DEVICE="cuda"  # or "cpu"

# 创建 checkpoint 文件夹
mkdir -p $CHECKPOINT_DIR

# Run training
python train_with_dinov2.py \
    --datapath $DATAPATH \
    --val_datapath $VAL_DATAPATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --image_size $IMAGE_SIZE \
    --encoder_size $ENCODER_SIZE \
    --unified_channels $UNIFIED_CHANNELS \
    --freeze_encoder \
    --adapter_at        $ADAPTER_AT \
    --adapter_reduction $ADAPTER_REDUCTION \
    --adapter_scale     $ADAPTER_SCALE \
    --lambda_dice $LAMBDA_DICE \
    --lambda_bce $LAMBDA_BCE \
    --lambda_coarse $LAMBDA_COARSE \
    --lambda_refined $LAMBDA_REFINED \
    --lambda_edge       $LAMBDA_EDGE \
    --edge_dilation     $EDGE_DILATION \
    --checkpoint_dir $CHECKPOINT_DIR \
    --save_freq $SAVE_FREQ \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"