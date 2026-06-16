#!/bin/bash
set -x

echo "Current script PID: $$"

python -u ./Transformer/main.py \
	--model_type 'Transformer_packets_feature_model' \
	--is_middle True \
	--feature_dim 7 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--train_num_samples 0 \
	--val_num_samples 0 \
	--median 1.389503 \
	--batch_size 256 \
	--epochs 30 \
	--dropout 0.1 \
	--lr 0.001 \
	--train_dir ./train_dir \
	--val_dir ./val_dir \
	--save_dir ./checkpoint_dir \
	--gpu_id 0 \
	--masked_idx -1
