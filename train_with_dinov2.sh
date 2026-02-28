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

# 损失权重
LAMBDA_DICE=1.0
LAMBDA_BCE=1.0
LAMBDA_COARSE=0.5
LAMBDA_REFINED=0.3

# Checkpoint
CHECKPOINT_DIR="checkpoint/E2Net_DINOv2"
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
    --lambda_dice $LAMBDA_DICE \
    --lambda_bce $LAMBDA_BCE \
    --lambda_coarse $LAMBDA_COARSE \
    --lambda_refined $LAMBDA_REFINED \
    --checkpoint_dir $CHECKPOINT_DIR \
    --save_freq $SAVE_FREQ \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"