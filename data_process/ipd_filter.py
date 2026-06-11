import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import torch
import sys

def ipd_check(root):
	for subdir, _, files in os.walk(root):
		for file in files:
			if file.endswith('.csv'):
				file_path = os.path.join(subdir, file)
				try:
					df = pd.read_csv(file_path, header=0)  # English note retained from the original workflow.
					if 'interval' not in df.columns:
						print(f"File {file_path} has no 'interval' column.")
					else:
						data = df['interval'].values
						if np.any((data == 0) | (data >= 1000)):
							print(f"File {file_path} contains interval == 0 or >= 1000.")
				except Exception as e:
					print(f"Error reading {file_path}: {e}")

def get_median(root):
	interval_diffs = []
	all_csv_files = []
	for subdir, _, files in os.walk(root):
		for file in files:
			if file.endswith('.csv'):
				file_path = os.path.join(subdir, file)
				all_csv_files.append(file_path)
	
	total_ipds = 0
	for csv_file in tqdm(all_csv_files, desc="Calculating median interval diff", mininterval=600):
		try:
			df = pd.read_csv(csv_file)
			if 'interval' not in df.columns or len(df) < 2:
				continue
			intervals = df['interval'].values
			diffs = np.abs(intervals[1:] - intervals[:-1])
			interval_diffs.extend(diffs.tolist())
			total_ipds += len(intervals)
			
		except:
			continue
	if not interval_diffs:
		raise ValueError("No valid interval differences found")
	median_diff = np.median(interval_diffs)
	print(f"total IPDs so far: {total_ipds}")
	print(f"Median of interval differences: {median_diff:.6f}")
if __name__ == "__main__":
	get_median("./test_dir")
