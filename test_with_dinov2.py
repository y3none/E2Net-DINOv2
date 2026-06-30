#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 E2Net + DINOv2 编码器模型，在测试集上生成预测结果。

【关键修复】
  预测图保存为「连续灰度概率图」(0-255)，不再做 0.5 二值化。
  原因：S-measure / weighted-F / MAE 直接在连续概率上运算；
        max-F / max-E 需在 256 个阈值上扫曲线取最大值。
        若保存二值图，阈值曲线退化为台阶，会系统性低估
        maxFm / maxEm / Sm 并使 MAE 失真——与 metrics.py / eval.py
        的设计假设以及其他 COD 方法的惯例(均存灰度图)不一致。
  如确需二值图(仅用于某些定性展示)，加 --binary 显式开启。

  内置 compute_metrics 已重写为直接调用项目的 metrics.py，
  与独立评测脚本 eval.py 完全同源同口径。
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from E2Net_dinov2 import E2Net_DINOv2
try:
    from dataset import Data, Config
except ImportError:
    print("Warning: dataset_dinov2.py not found, using original dataset.py")
    from dataset import Data, Config


def save_prediction(pred, save_path):
    """
    将预测结果保存为 PNG 图像
    参数:
        pred: [H, W] 预测图（值域 0-1）
        save_path: 保存路径
    """
    pred = (np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pred).save(save_path)


def test_dataset(model, test_loader, save_dir, device, binary=False):
    """
    在指定数据集上进行测试并保存预测结果
    参数:
        model: 已加载的 E2Net 模型
        test_loader: 测试数据加载器
        save_dir: 预测结果保存目录
        device: 运行设备（cuda / cpu）
        binary: 若 True 则按 0.5 阈值保存二值图（默认 False，保存连续灰度概率图）
    """
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    print(f"Testing on dataset, saving to {save_dir}"
          f"  [{'binary' if binary else 'grayscale prob'}]")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')

        for images, shapes, names in pbar:
            images = images.to(device, dtype=torch.float32)

            # 前向传播（仅使用最终输出 Y_final，已经过 Sigmoid，值域 [0,1]）
            _, _, Y_final = model(images)

            # 处理批次中的每张图像
            for i in range(Y_final.size(0)):
                pred = Y_final[i, 0].cpu().numpy()

                # 恢复至原始图像尺寸（线性插值，保持连续概率）
                H, W = shapes[0][i].item(), shapes[1][i].item()
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)

                # 默认保存连续灰度概率图；仅在显式要求时二值化
                if binary:
                    pred = (pred > 0.5).astype(np.float32)

                # 保存预测结果
                name = names[i]
                save_path = os.path.join(save_dir, name.replace('.jpg', '.png'))
                save_prediction(pred, save_path)

            pbar.set_postfix({'saved': len(names)})

    print(f"✓ Testing complete. Predictions saved to {save_dir}")


def compute_metrics(pred_dir, gt_dir):
    """
    计算标准评估指标，直接调用项目 metrics.py，与 eval.py 同源同口径。
    预测图与 GT 均以 0-255 灰度读入（metrics.py 内部自行 /255 并归一化）。

    返回:
        metrics: 含 Sm / wFm / MAE / maxEm / meanEm / adpEm / maxFm / meanFm / adpFm 的字典
                 （论文表格对应列：S_m=Sm, E_m^max=maxEm, F_m^w=wFm, M=MAE）
    """
    try:
        from metrics import Fmeasure_and_FNR, WeightedFmeasure, Smeasure, Emeasure, MAE
    except ImportError:
        print("Warning: metrics.py not found. Skipping metric computation.")
        return None

    FM  = Fmeasure_and_FNR()
    WFM = WeightedFmeasure()
    SM  = Smeasure()
    EM  = Emeasure()
    M   = MAE()

    pred_files = sorted(os.listdir(pred_dir))
    print("Computing metrics...")

    n_eval = 0
    for pred_file in tqdm(pred_files):
        gt_path = os.path.join(gt_dir, pred_file)
        if not os.path.exists(gt_path):
            continue

        # 注意：传入 0-255 uint8 灰度，切勿在此处再 /255（metrics.py 内部已处理）
        pred = cv2.imread(os.path.join(pred_dir, pred_file), cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_LINEAR)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)
        n_eval += 1

    if n_eval == 0:
        print("Warning: no matched pred/GT pairs found.")
        return None

    fm  = FM.get_results()[0]['fm']      # dict(adp=..., curve=...)
    em  = EM.get_results()['em']         # dict(adp=..., curve=...)
    wfm = WFM.get_results()['wfm']
    sm  = SM.get_results()['sm']
    mae = M.get_results()['mae']

    em_curve = em['curve']
    fm_curve = fm['curve']
    metrics = {
        'Sm':     float(sm),
        'wFm':    float(wfm),
        'MAE':    float(mae),
        'adpEm':  float(em['adp']),
        'meanEm': None if em_curve is None else float(em_curve.mean()),
        'maxEm':  None if em_curve is None else float(em_curve.max()),
        'adpFm':  float(fm['adp']),
        'meanFm': None if fm_curve is None else float(fm_curve.mean()),
        'maxFm':  None if fm_curve is None else float(fm_curve.max()),
    }
    return metrics


def _fmt(v):
    return '-' if v is None else f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser(description='Test E2Net with DINOv2')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--encoder_size', type=str, default='base',
                        choices=['small', 'base', 'large', 'giant'],
                        help='DINOv2 model size')
    parser.add_argument('--unified_channels', type=int, default=256,
                        help='Unified channel dimension')

    # 测试数据参数
    parser.add_argument('--test_datasets', type=str, nargs='+',
                        default=['CAMO', 'COD10K', 'CHAMELEON', 'NC4K'],
                        help='Test datasets to evaluate')
    parser.add_argument('--test_root', type=str, default='../dataset/TestDataset',
                        help='Root directory of test datasets')

    # 输出参数
    parser.add_argument('--save_dir', type=str, default='results/E2Net_DINOv2',
                        help='Directory to save predictions')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute evaluation metrics (in-script, same as eval.py)')
    parser.add_argument('--binary', action='store_true',
                        help='保存二值图(阈值0.5)。默认关闭——评测务必用连续灰度图。')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    # 图像尺寸（需与训练时一致；最佳模型为 518）
    parser.add_argument('--image_size', type=int, default=518,
                        help='Test image size (must match training, e.g. 518)')

    args = parser.parse_args()

    if args.binary:
        print("\n[!] --binary 已开启：将保存二值图。"
              "注意此模式不适用于 maxFm/maxEm/Sm/MAE 的标准评测。")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 创建模型
    print("\nLoading E2Net with DINOv2...")
    model = E2Net_DINOv2(
        encoder_size=args.encoder_size,
        freeze_encoder=True,
        unified_channels=args.unified_channels,
        # adapter_at=[]
        # adapter_at=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        adapter_at=[3, 6, 9, 11]
    )

    # 加载 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 E2Net + DINOv2 编码器模型，在测试集上生成预测结果。

【关键修复】
  预测图保存为「连续灰度概率图」(0-255)，不再做 0.5 二值化。
  原因：S-measure / weighted-F / MAE 直接在连续概率上运算；
        max-F / max-E 需在 256 个阈值上扫曲线取最大值。
        若保存二值图，阈值曲线退化为台阶，会系统性低估
        maxFm / maxEm / Sm 并使 MAE 失真——与 metrics.py / eval.py
        的设计假设以及其他 COD 方法的惯例(均存灰度图)不一致。
  如确需二值图(仅用于某些定性展示)，加 --binary 显式开启。

  内置 compute_metrics 已重写为直接调用项目的 metrics.py，
  与独立评测脚本 eval.py 完全同源同口径。
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from E2Net_dinov2 import E2Net_DINOv2
try:
    from dataset import Data, Config
except ImportError:
    print("Warning: dataset_dinov2.py not found, using original dataset.py")
    from dataset import Data, Config


def save_prediction(pred, save_path):
    """
    将预测结果保存为 PNG 图像
    参数:
        pred: [H, W] 预测图（值域 0-1）
        save_path: 保存路径
    """
    pred = (np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pred).save(save_path)


def test_dataset(model, test_loader, save_dir, device, binary=False):
    """
    在指定数据集上进行测试并保存预测结果
    参数:
        model: 已加载的 E2Net 模型
        test_loader: 测试数据加载器
        save_dir: 预测结果保存目录
        device: 运行设备（cuda / cpu）
        binary: 若 True 则按 0.5 阈值保存二值图（默认 False，保存连续灰度概率图）
    """
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    print(f"Testing on dataset, saving to {save_dir}"
          f"  [{'binary' if binary else 'grayscale prob'}]")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')

        for images, shapes, names in pbar:
            images = images.to(device, dtype=torch.float32)

            # 前向传播（仅使用最终输出 Y_final，已经过 Sigmoid，值域 [0,1]）
            _, _, Y_final = model(images)

            # 处理批次中的每张图像
            for i in range(Y_final.size(0)):
                pred = Y_final[i, 0].cpu().numpy()

                # 恢复至原始图像尺寸（线性插值，保持连续概率）
                H, W = shapes[0][i].item(), shapes[1][i].item()
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)

                # 默认保存连续灰度概率图；仅在显式要求时二值化
                if binary:
                    pred = (pred > 0.5).astype(np.float32)

                # 保存预测结果
                name = names[i]
                save_path = os.path.join(save_dir, name.replace('.jpg', '.png'))
                save_prediction(pred, save_path)

            pbar.set_postfix({'saved': len(names)})

    print(f"✓ Testing complete. Predictions saved to {save_dir}")


def compute_metrics(pred_dir, gt_dir):
    """
    计算标准评估指标，直接调用项目 metrics.py，与 eval.py 同源同口径。
    预测图与 GT 均以 0-255 灰度读入（metrics.py 内部自行 /255 并归一化）。

    返回:
        metrics: 含 Sm / wFm / MAE / maxEm / meanEm / adpEm / maxFm / meanFm / adpFm 的字典
                 （论文表格对应列：S_m=Sm, αE=adpEm, F_m^w=wFm, M=MAE）
    """
    try:
        from metrics import Fmeasure_and_FNR, WeightedFmeasure, Smeasure, Emeasure, MAE
    except ImportError:
        print("Warning: metrics.py not found. Skipping metric computation.")
        return None

    FM  = Fmeasure_and_FNR()
    WFM = WeightedFmeasure()
    SM  = Smeasure()
    EM  = Emeasure()
    M   = MAE()

    pred_files = sorted(os.listdir(pred_dir))
    print("Computing metrics...")

    n_eval = 0
    for pred_file in tqdm(pred_files):
        gt_path = os.path.join(gt_dir, pred_file)
        if not os.path.exists(gt_path):
            continue

        # 注意：传入 0-255 uint8 灰度，切勿在此处再 /255（metrics.py 内部已处理）
        pred = cv2.imread(os.path.join(pred_dir, pred_file), cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_LINEAR)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)
        n_eval += 1

    if n_eval == 0:
        print("Warning: no matched pred/GT pairs found.")
        return None

    fm  = FM.get_results()[0]['fm']      # dict(adp=..., curve=...)
    em  = EM.get_results()['em']         # dict(adp=..., curve=...)
    wfm = WFM.get_results()['wfm']
    sm  = SM.get_results()['sm']
    mae = M.get_results()['mae']

    em_curve = em['curve']
    fm_curve = fm['curve']
    metrics = {
        'Sm':     float(sm),
        'wFm':    float(wfm),
        'MAE':    float(mae),
        'adpEm':  float(em['adp']),
        'meanEm': None if em_curve is None else float(em_curve.mean()),
        'maxEm':  None if em_curve is None else float(em_curve.max()),
        'adpFm':  float(fm['adp']),
        'meanFm': None if fm_curve is None else float(fm_curve.mean()),
        'maxFm':  None if fm_curve is None else float(fm_curve.max()),
    }
    return metrics


def _fmt(v):
    return '-' if v is None else f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser(description='Test E2Net with DINOv2')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--encoder_size', type=str, default='base',
                        choices=['small', 'base', 'large', 'giant'],
                        help='DINOv2 model size')
    parser.add_argument('--unified_channels', type=int, default=256,
                        help='Unified channel dimension')

    # 测试数据参数
    parser.add_argument('--test_datasets', type=str, nargs='+',
                        default=['CAMO', 'COD10K', 'CHAMELEON', 'NC4K'],
                        help='Test datasets to evaluate')
    parser.add_argument('--test_root', type=str, default='../dataset/TestDataset',
                        help='Root directory of test datasets')

    # 输出参数
    parser.add_argument('--save_dir', type=str, default='results/E2Net_DINOv2',
                        help='Directory to save predictions')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute evaluation metrics (in-script, same as eval.py)')
    parser.add_argument('--binary', action='store_true',
                        help='保存二值图(阈值0.5)。默认关闭——评测务必用连续灰度图。')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    # 图像尺寸（需与训练时一致；最佳模型为 518）
    parser.add_argument('--image_size', type=int, default=518,
                        help='Test image size (must match training, default 518)')

    args = parser.parse_args()

    if args.binary:
        print("\n[!] --binary 已开启：将保存二值图。"
              "注意此模式不适用于 maxFm/maxEm/Sm/MAE 的标准评测。")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 创建模型
    print("\nLoading E2Net with DINOv2...")
    model = E2Net_DINOv2(
        encoder_size=args.encoder_size,
        freeze_encoder=True,
        unified_channels=args.unified_channels,
        # adapter_at=[]
        # adapter_at=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        adapter_at=[3, 6, 9, 11]
    )

    # 加载 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")

    # 开始测试各数据集
    print("\n" + "="*70)
    print("Testing on datasets")
    print("="*70)

    all_metrics = {}

    for dataset_name in args.test_datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        # 数据集路径
        test_path = os.path.join(args.test_root, dataset_name)

        if not os.path.exists(test_path):
            print(f"Warning: {test_path} does not exist, skipping...")
            continue

        # 创建测试数据加载器
        cfg_test = Config(
            datapath=test_path,
            mode='test',
            snapshot=None,
            batch_size=1,
            image_size=args.image_size
        )
        test_data = Data(cfg_test, 'E2Net')
        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )

        print(f"Test samples: {len(test_data)}")

        # 为当前数据集创建保存目录
        save_dir = os.path.join(args.save_dir, dataset_name)

        # 执行测试并保存预测
        test_dataset(model, test_loader, save_dir, device, binary=args.binary)

        # 若启用指标计算，则执行评估
        if args.compute_metrics:
            if args.binary:
                print("Warning: --binary 与标准评测口径不符，跳过 in-script metrics。")
            else:
                gt_dir = os.path.join(test_path, 'GT')
                if os.path.exists(gt_dir):
                    metrics = compute_metrics(save_dir, gt_dir)
                    if metrics:
                        all_metrics[dataset_name] = metrics

                        print(f"\nMetrics for {dataset_name}:")
                        print(f"  S_m (Sm)      : {_fmt(metrics['Sm'])}")
                        print(f"  αE  (adpEm)   : {_fmt(metrics['adpEm'])}")
                        print(f"  F_m^w (wFm)   : {_fmt(metrics['wFm'])}")
                        print(f"  M   (MAE)     : {_fmt(metrics['MAE'])}")
                        print(f"  -- 其它 --")
                        print(f"  meanEm: {_fmt(metrics['meanEm'])}  maxEm: {_fmt(metrics['maxEm'])}")
                        print(f"  maxFm : {_fmt(metrics['maxFm'])}  "
                              f"meanFm: {_fmt(metrics['meanFm'])}  adpFm: {_fmt(metrics['adpFm'])}")
                else:
                    print(f"Warning: Ground truth not found at {gt_dir}")

    # 总结
    print("\n" + "="*70)
    print("Testing Summary")
    print("="*70)

    if all_metrics:
        print("\nResults (paper columns: S_m, αE, F_m^w, M):")
        print(f"{'Dataset':<12} {'Sm':<8} {'adpEm':<8} {'wFm':<8} {'MAE':<8}")
        print("-" * 70)
        for dataset, m in all_metrics.items():
            print(f"{dataset:<12} {_fmt(m['Sm']):<8} {_fmt(m['adpEm']):<8} "
                  f"{_fmt(m['wFm']):<8} {_fmt(m['MAE']):<8}")

        # 保存结果
        results_file = os.path.join(args.save_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write("E2Net with DINOv2 - Test Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Dataset':<12} {'Sm':<8} {'adpEm':<8} {'wFm':<8} {'MAE':<8} "
                    f"{'meanEm':<8} {'maxEm':<8} {'maxFm':<8} {'meanFm':<8} {'adpFm':<8}\n")
            f.write("-"*70 + "\n")
            for dataset, m in all_metrics.items():
                f.write(f"{dataset:<12} {_fmt(m['Sm']):<8} {_fmt(m['adpEm']):<8} "
                        f"{_fmt(m['wFm']):<8} {_fmt(m['MAE']):<8} "
                        f"{_fmt(m['meanEm']):<8} {_fmt(m['maxEm']):<8} "
                        f"{_fmt(m['maxFm']):<8} {_fmt(m['meanFm']):<8} {_fmt(m['adpFm']):<8}\n")

        print(f"\n✓ Results saved to {results_file}")

    print(f"\n✓ All predictions saved to {args.save_dir}")
    print("\nTesting completed!")


if __name__ == '__main__':
    main()
    print("✓ Model loaded successfully")

    # 开始测试各数据集
    print("\n" + "="*70)
    print("Testing on datasets")
    print("="*70)

    all_metrics = {}

    for dataset_name in args.test_datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        # 数据集路径
        test_path = os.path.join(args.test_root, dataset_name)

        if not os.path.exists(test_path):
            print(f"Warning: {test_path} does not exist, skipping...")
            continue

        # 创建测试数据加载器
        cfg_test = Config(
            datapath=test_path,
            mode='test',
            snapshot=None,
            batch_size=1,
            image_size=args.image_size
        )
        test_data = Data(cfg_test, 'E2Net')
        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )

        print(f"Test samples: {len(test_data)}")

        # 为当前数据集创建保存目录
        save_dir = os.path.join(args.save_dir, dataset_name)

        # 执行测试并保存预测
        test_dataset(model, test_loader, save_dir, device, binary=args.binary)

        # 若启用指标计算，则执行评估
        if args.compute_metrics:
            if args.binary:
                print("Warning: --binary 与标准评测口径不符，跳过 in-script metrics。")
            else:
                gt_dir = os.path.join(test_path, 'GT')
                if os.path.exists(gt_dir):
                    metrics = compute_metrics(save_dir, gt_dir)
                    if metrics:
                        all_metrics[dataset_name] = metrics

                        print(f"\nMetrics for {dataset_name}:")
                        print(f"  S_m (Sm)      : {_fmt(metrics['Sm'])}")
                        print(f"  E_m^max(maxEm): {_fmt(metrics['maxEm'])}")
                        print(f"  F_m^w (wFm)   : {_fmt(metrics['wFm'])}")
                        print(f"  M   (MAE)     : {_fmt(metrics['MAE'])}")
                        print(f"  -- 其它 --")
                        print(f"  meanEm: {_fmt(metrics['meanEm'])}  adpEm: {_fmt(metrics['adpEm'])}")
                        print(f"  maxFm : {_fmt(metrics['maxFm'])}  "
                              f"meanFm: {_fmt(metrics['meanFm'])}  adpFm: {_fmt(metrics['adpFm'])}")
                else:
                    print(f"Warning: Ground truth not found at {gt_dir}")

    # 总结
    print("\n" + "="*70)
    print("Testing Summary")
    print("="*70)

    if all_metrics:
        print("\nResults (paper columns: S_m, E_m^max, F_m^w, M):")
        print(f"{'Dataset':<12} {'Sm':<8} {'maxEm':<8} {'wFm':<8} {'MAE':<8}")
        print("-" * 70)
        for dataset, m in all_metrics.items():
            print(f"{dataset:<12} {_fmt(m['Sm']):<8} {_fmt(m['maxEm']):<8} "
                  f"{_fmt(m['wFm']):<8} {_fmt(m['MAE']):<8}")

        # 保存结果
        results_file = os.path.join(args.save_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write("E2Net with DINOv2 - Test Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Dataset':<12} {'Sm':<8} {'maxEm':<8} {'wFm':<8} {'MAE':<8} "
                    f"{'meanEm':<8} {'adpEm':<8} {'maxFm':<8} {'meanFm':<8} {'adpFm':<8}\n")
            f.write("-"*70 + "\n")
            for dataset, m in all_metrics.items():
                f.write(f"{dataset:<12} {_fmt(m['Sm']):<8} {_fmt(m['maxEm']):<8} "
                        f"{_fmt(m['wFm']):<8} {_fmt(m['MAE']):<8} "
                        f"{_fmt(m['meanEm']):<8} {_fmt(m['adpEm']):<8} "
                        f"{_fmt(m['maxFm']):<8} {_fmt(m['meanFm']):<8} {_fmt(m['adpFm']):<8}\n")

        print(f"\n✓ Results saved to {results_file}")

    print(f"\n✓ All predictions saved to {args.save_dir}")
    print("\nTesting completed!")


if __name__ == '__main__':
    main()