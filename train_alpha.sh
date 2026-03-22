#!/bin/bash
# 方案 α — 无验证集训练，对标 CamoFormer
# best model 保存标准：训练 loss 最低；训练结束额外保存 final 权重

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "============================================================"
echo "方案 α — 无验证集（对标 CamoFormer）"
echo "============================================================"

# ── 数据（只需训练集，无 val）────────────────────────────────────────────
DATAPATH="../dataset/TrainDataset"

# ── 训练 ─────────────────────────────────────────────────────────────────
BATCH_SIZE=4
EPOCHS=100          # 固定轮数，训练结束取最终权重
LR=0.0001
IMAGE_SIZE=518

# ── 模型 ─────────────────────────────────────────────────────────────────
ENCODER_SIZE="base"
UNIFIED_CHANNELS=256

# ── 损失权重 ──────────────────────────────────────────────────────────────
LAMBDA_DICE=1.0
LAMBDA_BCE=1.0
# LAMBDA_AUX=0.3  # 辅助损失总权重（替代之前的 coarse/refined/iou 三个参数）
LAMBDA_IOU=1.0      # 新增 IoU Loss（0=关闭）
LAMBDA_COARSE=0.5
LAMBDA_REFINED=0.3

# ── Checkpoint ────────────────────────────────────────────────────────────
CHECKPOINT_DIR="checkpoint/E2Net_dinov2_alpha_13"
SAVE_FREQ=10        # 每 10 epoch 保存定期 checkpoint
DEVICE="cuda"

mkdir -p $CHECKPOINT_DIR

echo ""
echo "训练集   : ${DATAPATH}"
echo "验证集   : 无（全量数据用于训练）"
echo "保存标准 : 训练 loss 最低 → best；第 ${EPOCHS} epoch → final"
echo ""

python train_alpha.py \
    --datapath          $DATAPATH          \
    --batch_size        $BATCH_SIZE        \
    --epochs            $EPOCHS            \
    --lr                $LR                \
    --image_size        $IMAGE_SIZE        \
    --encoder_size      $ENCODER_SIZE      \
    --unified_channels  $UNIFIED_CHANNELS  \
    --freeze_encoder                       \
    --lambda_dice       $LAMBDA_DICE       \
    --lambda_bce        $LAMBDA_BCE        \
    --lambda_iou        $LAMBDA_IOU        \
    --lambda_coarse     $LAMBDA_COARSE        \
    --lambda_refined        $LAMBDA_REFINED        \
    --checkpoint_dir    $CHECKPOINT_DIR    \
    --save_freq         $SAVE_FREQ         \
    --device            $DEVICE

echo ""
echo "============================================================"
echo "Training completed!"
echo "  best  → ${CHECKPOINT_DIR}/E2Net_alpha_best.pth"
echo "  final → ${CHECKPOINT_DIR}/E2Net_alpha_final.pth"
echo "============================================================"
