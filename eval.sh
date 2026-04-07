python ./evaltools/eval.py   \
    --model  'E2Net_DINOv2' \
    --GT_root  './dataset/TestDataset' \
    --pred_root './files/results/E2Net_dinov2_alpha_18/' \
    --record_path './files/results/E2Net_dinov2_alpha_18/eval_record.txt' \
    --BR 'on'