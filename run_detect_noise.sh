#!/bin/bash
set -x

echo "当前脚本 PID: $$"

python -u ./data_process/nctc_detect_noise.py \
	--model_type 'Transformer_packets_feature_model' \
	--feature_dim 7 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--checkpoint_path './results_2026/Transformer_6_packets_feature_is_middleTrue_train_num0_val_num0_epochs30/epoch_9.pth' \
	--is_middle True \
	--dropout 0.1 \
	--num_samples 0 \
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500_5_0.1' \
	--nctc_root './data_emerged/noise_data_5_0.1' \
	--median 1.389503 \
	--batch_size 1024 \
	--gpu_id 2 \
	--masked_idx -1


python -u ./data_process/nctc_detect_noise.py \
	--model_type 'Transformer_packets_feature_model' \
	--feature_dim 7 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--checkpoint_path './results_2026/Transformer_6_packets_feature_is_middleTrue_train_num0_val_num0_epochs30/epoch_9.pth' \
	--is_middle True \
	--dropout 0.1 \
	--num_samples 0 \
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500_5_0.2' \
	--nctc_root './data_emerged/noise_data_5_0.2' \
	--median 1.389503 \
	--batch_size 1024 \
	--gpu_id 2 \
	--masked_idx -1


python -u ./data_process/nctc_detect_noise.py \
	--model_type 'Transformer_packets_feature_model' \
	--feature_dim 7 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--checkpoint_path './results_2026/Transformer_6_packets_feature_is_middleTrue_train_num0_val_num0_epochs30/epoch_9.pth' \
	--is_middle True \
	--dropout 0.1 \
	--num_samples 0 \
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500_5_0.4' \
	--nctc_root './data_emerged/noise_data_5_0.4' \
	--median 1.389503 \
	--batch_size 1024 \
	--gpu_id 2 \
	--masked_idx -1

python -u ./data_process/nctc_detect_noise.py \
	--model_type 'Transformer_packets_feature_model' \
	--feature_dim 7 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--checkpoint_path './results_2026/Transformer_6_packets_feature_is_middleTrue_train_num0_val_num0_epochs30/epoch_9.pth' \
	--is_middle True \
	--dropout 0.1 \
	--num_samples 0 \
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500_10_0.1' \
	--nctc_root './data_emerged/noise_data_10_0.1' \
	--median 1.389503 \
	--batch_size 1024 \
	--gpu_id 2 \
	--masked_idx -1

python -u ./data_process/nctc_detect_noise.py \
	--model_type 'Transformer_packets_feature_model' \
	--feature_dim 7 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--checkpoint_path './results_2026/Transformer_6_packets_feature_is_middleTrue_train_num0_val_num0_epochs30/epoch_9.pth' \
	--is_middle True \
	--dropout 0.1 \
	--num_samples 0 \
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500_10_0.2' \
	--nctc_root './data_emerged/noise_data_10_0.2' \
	--median 1.389503 \
	--batch_size 1024 \
	--gpu_id 2 \
	--masked_idx -1

python -u ./data_process/nctc_detect_noise.py \
	--model_type 'Transformer_packets_feature_model' \
	--feature_dim 7 \
	--d_model 100 \
	--nhead 4 \
	--num_layers 2 \
	--dim_feedforward 256 \
	--seq_len 8 \
	--checkpoint_path './results_2026/Transformer_6_packets_feature_is_middleTrue_train_num0_val_num0_epochs30/epoch_9.pth' \
	--is_middle True \
	--dropout 0.1 \
	--num_samples 0 \
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500_10_0.4' \
	--nctc_root './data_emerged/noise_data_10_0.4' \
	--median 1.389503 \
	--batch_size 1024 \
	--gpu_id 2 \
	--masked_idx -1