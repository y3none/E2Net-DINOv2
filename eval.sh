python ./evaltools/eval.py   \
    --model  'E2Net_DINOv2' \
    --GT_root  './dataset/TestDataset' \
    --pred_root './files/results/E2Net_dinov2_alpha_18new_final_grey' \
    --record_path './files/results/E2Net_dinov2_alpha_18new_final_grey/eval_record.txt' \
    --BR 'on'