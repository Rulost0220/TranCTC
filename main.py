import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os
import argparse
import sys
from model import *
# from model_ori import IntervalClassifierTransformer, SequenceFeaturePredictorTransformer, Transformer_Feature_extract_and_Predict, IntervalDataset, train_val
from torch.utils.data import WeightedRandomSampler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

def main(args):
	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
	print(f"Using Device{device}")
	model_registry = {
		"Transformer_packets_feature_model": (Transformer_packets_feature_model, train_val),
		"Transformer_ipd_model": (Transformer_ipd_model, train_val_ipd_transformer),
	}
	print('-----------------------Loading Training Dataset-----------------------')

	train_loader = get_dataloader(args.model_type, args.train_dir, args.feature_dim, args.seq_len, args.median, args.train_num_samples, args.batch_size, args.is_middle, args.masked_idx)
	val_loader = get_dataloader(args.model_type, args.val_dir, args.feature_dim, args.seq_len, args.median, args.val_num_samples, args.batch_size, args.is_middle, args.masked_idx)
	print('------------------------Training Dataset Loaded-----------------------')

	print('-----------------------------Initial Model----------------------------')
	ModelClass, train_fun = model_registry[args.model_type] 
	model = ModelClass(
		discretized_size=16,  # 假设离散化后有 100 个类别
		feature_dim=args.feature_dim,
		d_model=args.d_model,
		nhead=args.nhead,
		num_layers=args.num_layers,
		dim_feedforward=args.dim_feedforward,
		dropout=args.dropout,
		is_middle=args.is_middle
	).to(device)
	if ModelClass == Transformer_packets_feature_model:
		model.masked_idx = args.masked_idx
		print(f"masked_idx = {model.masked_idx}")

	print('Initial Finished！')

	print('---------------------Defining Criterion and Optimizer-----------------')
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
	# # Warmup
	# warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
	print("--------------------Criterion and Optimizer Defined-------------------")

	print("----------------------------Start Training----------------------------")
	train_fun(
		model,
		train_loader,
		val_loader,
		criterion,
		optimizer,
		epochs=args.epochs,
		save_dir=args.save_dir,
		device=device
	)
	print("-------------------------Training Finished---------------------------")

def get_args():
	parser = argparse.ArgumentParser(description="模型训练参数")
	
	# 添加常用参数
	parser.add_argument('--model_type', type=str, default='base', help='选择模型架构')
	parser.add_argument('--is_middle', type=str2bool, help='是否结合上下文信息')
	parser.add_argument('--feature_dim', type=int, default=480, help='特征维度')
	parser.add_argument('--dim_feedforward', type=int, default=256, help='前馈网络维度')
	parser.add_argument('--seq_len', type=int, default=9, help='序列长度')
	parser.add_argument('--d_model', type=int, default=256, help='模型维度')
	parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
	parser.add_argument('--num_layers', type=int, default=3, help='Transformer 层数')

	parser.add_argument('--train_num_samples', type=int, default=800000, help='训练样本数')
	parser.add_argument('--val_num_samples', type=int, default=750000, help='验证样本数')
	parser.add_argument('--median', type=float)
	parser.add_argument('--batch_size', type=int, default=64, help='批大小')
	parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
	parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 概率')
	parser.add_argument('--lr', type=float, default=0.001, help='学习率')

	parser.add_argument('--train_dir', type=str, default='data/npy_data/train/', help='训练数据目录')
	parser.add_argument('--val_dir', type=str, default='data/npy_data/val/', help='验证数据目录')
	parser.add_argument('--save_dir', type=str, default=None, help='模型训练结果保存目录')
	# 新增 GPU ID 参数
	parser.add_argument('--gpu_id', type=int, default=0, help='使用的 GPU ID（例如 0 表示 cuda:0）')
	parser.add_argument('--masked_idx', type=int, default=-1, help='masked 特征')
	args = parser.parse_args()
	# if args.save_dir is None:
	# 	args.save_dir = os.path.join(
	# 		'Transformer/model_pth',
	# 		f'{args.model_type}_all_feature_is_middle{args.is_middle}_epochs{args.epochs}'
	# 	)
	if args.masked_idx != -1 and args.save_dir is not None:
		args.save_dir = f'{args.save_dir}_masked_{args.masked_idx}'
		args.feature_dim = args.feature_dim - 1

	print("Model Checkpoint Save path",args.save_dir)
	# 确保保存目录存在
	os.makedirs(args.save_dir, exist_ok=True)
	return args

if __name__ == '__main__':
	args = get_args()
	main(args)