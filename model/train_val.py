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
		discretized_size=16,  
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
	parser = argparse.ArgumentParser(description="Model training parameters")
	
	parser.add_argument('--model_type', type=str, default='base', help='Select model architecture')
	parser.add_argument('--is_middle', type=str2bool, help='Whether to use context information')
	parser.add_argument('--feature_dim', type=int, default=480, help='Feature dimension')
	parser.add_argument('--dim_feedforward', type=int, default=256, help='Feed-forward network dimension')
	parser.add_argument('--seq_len', type=int, default=9, help='Sequence length')
	parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
	parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
	parser.add_argument('--num_layers', type=int, default=3, help='Number of Transformer layers')

	parser.add_argument('--train_num_samples', type=int, default=800000, help='Number of training samples')
	parser.add_argument('--val_num_samples', type=int, default=750000, help='Number of validation samples')
	parser.add_argument('--median', type=float)
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
	parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
	parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

	parser.add_argument('--train_dir', type=str, default='./train_dir', help='Training data directory')
	parser.add_argument('--val_dir', type=str, default='./val_dir', help='Validation data directory')
	parser.add_argument('--save_dir', type=str, default=None, help='Directory for saved training outputs')
	parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use, for example 0 means cuda:0')
	parser.add_argument('--masked_idx', type=int, default=-1, help='masked feature')
	args = parser.parse_args()

	if args.masked_idx != -1 and args.save_dir is not None:
		args.save_dir = f'{args.save_dir}_masked_{args.masked_idx}'
		args.feature_dim = args.feature_dim - 1

	print("Model Checkpoint Save path",args.save_dir)
	os.makedirs(args.save_dir, exist_ok=True)
	return args

if __name__ == '__main__':
	args = get_args()
	main(args)
