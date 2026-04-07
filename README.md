# E2Net with DINOv2
### 训练
辅助损失 IOU + COARSE + REFINED：`bash train_alpha.sh`  
辅助损失 AUX：`bash train_alpha_newloss.sh`  
### 测试
`bash test_with_dinov2.sh`  
### 评估
`bash eval.sh`  
### Performance
![](/figs/E2Net_performance_04.png)