#!/bin/bash
set -x

echo "当前脚本 PID: $$"


# python -u ./data_process/nctc_detect.py \
# 	--model_type 'Transformer_packets_feature_model' \
# 	--feature_dim 16 \
# 	--d_model 100 \
# 	--nhead 4 \
# 	--num_layers 2 \
# 	--dim_feedforward 256 \
# 	--seq_len 8 \
# 	--checkpoint_path './Transformer/model_pth/Transformer_packets_feature_model_all_feature_is_middleFalse_epochs30/epoch_3.pth' \
# 	--is_middle False \
# 	--dropout 0.1 \
# 	--num_samples 0 \
# 	--normal_dir './data_emerged/train_test_flow/reference_csv' \
# 	--nctc_root './data_emerged/nctc_data' \
# 	--median 1.389503 \
# 	--batch_size 128 \
# 	--gpu_id 2


# v4:1.1661052703857155
# v6:1.389503

# python -u ./data_process/nctc_detect.py \
# 	--model_type GAS \
# 	--feature_dim 16 \
# 	--d_model 100 \
# 	--nhead 4 \
# 	--num_layers 2 \
# 	--dim_feedforward 256 \
# 	--seq_len 8 \
# 	--checkpoint_path ./GAS/v4_gas_model_checkpoint/epoch_50.pth \
# 	--is_middle False \
# 	--dropout 0.1 \
# 	--num_samples 0 \
# 	--normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
# 	--nctc_root './data_emerged/nctc_data' \
# 	--median 1.1661052703857155 \
# 	--batch_size 128 \
# 	--gpu_id 2

#
# ## 检测Transformer_ipd_model（Not Middle）
#  python -u ./data_process/nctc_detect.py \
#  	--model_type Transformer_ipd_model \
#  	--feature_dim 16 \
#  	--d_model 100 \
#  	--nhead 4 \
#  	--num_layers 2 \
#  	--dim_feedforward 256 \
#  	--seq_len 8 \
#  	--checkpoint_path ./Transformer/model_pth/Transformer_ipd_model_is_middleFalse_train_num0_val_num0_epochs30/best_epoch.pth \
#  	--is_middle False \
#  	--dropout 0.1 \
#  	--num_samples 0 \
#  	--normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
#  	--nctc_root './lnctc_data/' \
#  	--median 1.389503 \
#  	--batch_size 128 \
#  	--gpu_id 0

# ## 检测Transformer_ipd_model（Middle）
#  python -u ./data_process/nctc_detect.py \
#  	--model_type Transformer_ipd_model \
#  	--feature_dim 16 \
#  	--d_model 100 \
#  	--nhead 4 \
#  	--num_layers 2 \
#  	--dim_feedforward 256 \
#  	--seq_len 8 \
#  	--checkpoint_path ./Transformer/model_pth/Transformer_ipd_model_is_middleTrue_train_num0_val_num0_epochs30/best_epoch.pth \
#  	--is_middle True \
#  	--dropout 0.1 \
#  	--num_samples 0 \
#  	--normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
#  	--nctc_root './lnctc_data/' \
#  	--median 1.389503 \
#  	--batch_size 128 \
#  	--gpu_id 0

# # # 检测 GAS(trained_on_v4)
# python -u ./data_process/nctc_detect.py \
# 	--model_type GAS \
# 	--feature_dim 16 \
# 	--d_model 100 \
# 	--nhead 4 \
# 	--num_layers 2 \
# 	--dim_feedforward 256 \
# 	--seq_len 8 \
# 	--checkpoint_path ./GAS/v4_gas_model_checkpoint/epoch_50.pth \
# 	--is_middle False \
# 	--dropout 0.1 \
# 	--num_samples 0 \
# 	--normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
# 	--nctc_root './data_emerged/nctc_data_2' \
# 	--median 1.389503 \
# 	--batch_size 128 \
# 	--gpu_id 1


# # # 检测 GAS(trained_on_v6)
# python -u ./data_process/nctc_detect.py \
# 	--model_type GAS \
# 	--feature_dim 16 \
# 	--d_model 100 \
# 	--nhead 4 \
# 	--num_layers 2 \
# 	--dim_feedforward 256 \
# 	--seq_len 8 \
# 	--checkpoint_path ./GAS/v6_gas_model_checkpoint/epoch_50.pth \
# 	--is_middle False \
# 	--dropout 0.1 \
# 	--num_samples 0 \
# 	--normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
# 	--nctc_root './lnctc_data' \
# 	--median 1.389503 \
# 	--batch_size 128 \
# 	--gpu_id 0

## 检测其他各模型
# python -u ./data_process/nctc_detect.py \
# 	--model_type Other_Models \
# 	--feature_dim 16 \
# 	--d_model 100 \
# 	--nhead 4 \
# 	--num_layers 2 \
# 	--dim_feedforward 256 \
# 	--seq_len 8 \
# 	--checkpoint_path ./GAS/v4_gas_model_checkpoint/epoch_50.pth \
# 	--is_middle False \
# 	--dropout 0.1 \
# 	--num_samples 0 \
# 	--normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
# 	--nctc_root './lnctc_data' \
# 	--median 1.389503 \
# 	--batch_size 128 \
# 	--gpu_id 0

# 检测 Transformer Packet feature(Middle)
python -u ./data_process/nctc_detect.py \
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
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500' \
	--nctc_root './data_emerged/75_flows/test_normal_flow/' \
	--median 1.389503 \
	--batch_size 256 \
	--gpu_id 0 \
	--masked_idx -1 \
	--flow_num 7485

python -u ./data_process/nctc_detect.py \
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
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500' \
	--nctc_root './data_emerged/75_flows/test_normal_flow/' \
	--median 1.389503 \
	--batch_size 256 \
	--gpu_id 1 \
	--masked_idx -1 \
	--flow_num 7462

python -u ./data_process/nctc_detect.py \
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
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500' \
	--nctc_root './data_emerged/75_flows/test_normal_flow/' \
	--median 1.389503 \
	--batch_size 256 \
	--gpu_id 2 \
	--masked_idx -1 \
	--flow_num 7425

python -u ./data_process/nctc_detect.py \
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
	--normal_dir './data_emerged/train_test_flow/reference_csv_v6_7500' \
	--nctc_root './data_emerged/75_flows/test_normal_flow/' \
	--median 1.389503 \
	--batch_size 256 \
	--gpu_id 0 \
	--masked_idx -1 \
	--flow_num 7352
# --normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
# --nctc_root './data_emerged/nctc_data_2' \
# 70分位数:--median 4.84132767 \
# 	--median 1.389503 \
# --checkpoint_path './results_2026/Transformer_6_packets_feature_is_middleTrue_train_num0_val_num0_epochs30/epoch_9.pth' \
# --checkpoint_path './Transformer/model_pth/Transformer_packets_feature_model_all_feature_is_middleTrue_epochs30/epoch_18.pth' \ 




# ## 检测 Transformer Packet feature(Not Middle)
# python -u ./data_process/nctc_detect.py \
# 	--model_type 'Transformer_packets_feature_model' \
# 	--feature_dim 16 \
# 	--d_model 100 \
# 	--nhead 4 \
# 	--num_layers 2 \
# 	--dim_feedforward 256 \
# 	--seq_len 8 \
# 	--checkpoint_path './Transformer/model_pth/Transformer_packets_feature_model_all_feature_is_middleFalse_epochs30/epoch_5.pth' \
# 	--is_middle False \
# 	--dropout 0.1 \
# 	--num_samples 0 \
# 	--normal_dir './data_emerged/train_test_flow/reference_csv_v6' \
#  	--nctc_root './lnctc_data/' \
# 	--median 1.389503 \
# 	--batch_size 512 \
# 	--gpu_id 0
