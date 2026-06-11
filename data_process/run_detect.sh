#!/bin/bash
set -x

echo "Current script PID: $$"

python -u ./data_process/nctc_detect.py \
	--model_type 'Transformer_packets_feature_model' \
	--feature_dim 7 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--checkpoint_path './checkpoint_dir/epoch.pth' \
	--is_middle True \
	--dropout 0.1 \
	--num_samples 0 \
	--normal_dir './normal_dir' \
	--nctc_root './nctc_dir' \
	--median 1.389503 \
	--batch_size 256 \
	--gpu_id 0 \
	--masked_idx -1 \
	--flow_num 0 \
	--reference_limit 0 \
	--low_q 0 \
	--high_q 99
