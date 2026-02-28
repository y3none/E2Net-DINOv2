#!/bin/bash
# 测试 E2Net with DINOv2

# 设置环境变量
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "============================================================"
echo "Testing E2Net with DINOv2 Encoder"
echo "============================================================"

# 配置
CHECKPOINT="checkpoint/E2Net_DINOv2/E2Net_DINOv2_best.pth"
TEST_ROOT="../dataset/TestDataset"
SAVE_DIR="results/E2Net_DINOv2"

# 测试数据集
TEST_DATASETS="CAMO COD10K CHAMELEON NC4K"

# Device
DEVICE="cuda"  # or "cpu"

# Run testing
python test_with_dinov2.py \
    --checkpoint $CHECKPOINT \
    --test_datasets $TEST_DATASETS \
    --test_root $TEST_ROOT \
    --save_dir $SAVE_DIR \
    --compute_metrics \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Testing completed!"
echo "Results saved to: $SAVE_DIR"
echo "============================================================"