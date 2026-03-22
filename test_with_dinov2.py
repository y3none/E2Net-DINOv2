#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 E2Net + DINOv2 编码器模型
在测试集上评估并生成预测结果
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
    pred = (pred * 255).astype(np.uint8)
    Image.fromarray(pred).save(save_path)


def test_dataset(model, test_loader, save_dir, device):
    """
    在指定数据集上进行测试并保存预测结果
    参数:
        model: 已加载的 E2Net 模型
        test_loader: 测试数据加载器
        save_dir: 预测结果保存目录
        device: 运行设备（cuda / cpu）
    """
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Testing on dataset, saving to {save_dir}")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, shapes, names in pbar:
            images = images.to(device, dtype=torch.float32)
            
            # 前向传播（仅使用最终输出 Y_final）
            _, _, Y_final = model(images)
            
            # 处理批次中的每张图像
            for i in range(Y_final.size(0)):
                pred = Y_final[i, 0].cpu().numpy()
                
                # 恢复至原始图像尺寸
                H, W = shapes[0][i].item(), shapes[1][i].item()
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
                
                # 二值化（阈值 0.5）
                pred = (pred > 0.5).astype(np.float32)
                
                # 保存预测结果
                name = names[i]
                save_path = os.path.join(save_dir, name.replace('.jpg', '.png'))
                save_prediction(pred, save_path)
            
            pbar.set_postfix({'saved': len(names)})
    
    print(f"✓ Testing complete. Predictions saved to {save_dir}")


def compute_metrics(pred_dir, gt_dir):
    """
    计算标准评估指标（MAE、F-measure、S-measure 等）
    参数:
        pred_dir: 预测结果目录
        gt_dir: 真实标签（GT）目录
    返回:
        metrics: 包含各项指标的字典
    """
    try:
        # 导入评估指标函数
        from eval_metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm
    except ImportError:
        print("Warning: eval_metrics not found. Skipping metric computation.")
        print("Please implement evaluation metrics or use external evaluation tools.")
        return None
    
    pred_files = sorted(os.listdir(pred_dir))
    
    mae_list = []
    fm_list = []
    sm_list = []
    em_list = []
    wfm_list = []
    
    print("Computing metrics...")
    
    for pred_file in tqdm(pred_files):
        # 加载预测图
        pred_path = os.path.join(pred_dir, pred_file)
        pred = np.array(Image.open(pred_path).convert('L')) / 255.0
        
        # 加载真实标签
        gt_path = os.path.join(gt_dir, pred_file)
        if not os.path.exists(gt_path):
            continue
        gt = np.array(Image.open(gt_path).convert('L')) / 255.0
        
        # 计算各项指标
        mae_list.append(cal_mae(pred, gt))
        fm_list.append(cal_fm(pred, gt))
        sm_list.append(cal_sm(pred, gt))
        em_list.append(cal_em(pred, gt))
        wfm_list.append(cal_wfm(pred, gt))
    
    metrics = {
        'MAE': np.mean(mae_list),
        'maxFm': np.max(fm_list),
        'avgFm': np.mean(fm_list),
        'Sm': np.mean(sm_list),
        'maxEm': np.max(em_list),
        'avgEm': np.mean(em_list),
        'wFm': np.mean(wfm_list)
    }
    
    return metrics


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
                        help='Compute evaluation metrics')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 创建模型
    print("\nLoading E2Net with DINOv2...")
    model = E2Net_DINOv2(
        encoder_size=args.encoder_size,
        freeze_encoder=True,
        unified_channels=args.unified_channels
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
            batch_size=1
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
        test_dataset(model, test_loader, save_dir, device)
        
        # 若启用指标计算，则执行评估
        if args.compute_metrics:
            gt_dir = os.path.join(test_path, 'GT')
            if os.path.exists(gt_dir):
                metrics = compute_metrics(save_dir, gt_dir)
                if metrics:
                    all_metrics[dataset_name] = metrics
                    
                    print(f"\nMetrics for {dataset_name}:")
                    print(f"  MAE: {metrics['MAE']:.4f}")
                    print(f"  maxFm: {metrics['maxFm']:.4f}")
                    print(f"  avgFm: {metrics['avgFm']:.4f}")
                    print(f"  Sm: {metrics['Sm']:.4f}")
                    print(f"  maxEm: {metrics['maxEm']:.4f}")
                    print(f"  avgEm: {metrics['avgEm']:.4f}")
                    print(f"  wFm: {metrics['wFm']:.4f}")
            else:
                print(f"Warning: Ground truth not found at {gt_dir}")
    
    # 总结
    print("\n" + "="*70)
    print("Testing Summary")
    print("="*70)
    
    if all_metrics:
        print("\nResults:")
        print(f"{'Dataset':<15} {'MAE':<8} {'maxFm':<8} {'Sm':<8} {'maxEm':<8}")
        print("-" * 70)
        for dataset, metrics in all_metrics.items():
            print(f"{dataset:<15} {metrics['MAE']:<8.4f} {metrics['maxFm']:<8.4f} "
                  f"{metrics['Sm']:<8.4f} {metrics['maxEm']:<8.4f}")
        
        # 保存结果
        results_file = os.path.join(args.save_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write("E2Net with DINOv2 - Test Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Dataset':<15} {'MAE':<8} {'maxFm':<8} {'avgFm':<8} {'Sm':<8} "
                   f"{'maxEm':<8} {'avgEm':<8} {'wFm':<8}\n")
            f.write("-"*70 + "\n")
            for dataset, metrics in all_metrics.items():
                f.write(f"{dataset:<15} {metrics['MAE']:<8.4f} {metrics['maxFm']:<8.4f} "
                       f"{metrics['avgFm']:<8.4f} {metrics['Sm']:<8.4f} "
                       f"{metrics['maxEm']:<8.4f} {metrics['avgEm']:<8.4f} "
                       f"{metrics['wFm']:<8.4f}\n")
        
        print(f"\n✓ Results saved to {results_file}")
    
    print(f"\n✓ All predictions saved to {args.save_dir}")
    print("\nTesting completed!")


if __name__ == '__main__':
    main()