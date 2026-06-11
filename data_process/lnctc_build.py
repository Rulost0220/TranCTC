import math
import random
import numpy as np
import pandas as pd
from scipy.stats import gamma, exponweib, pareto, lognorm, weibull_min
from sklearn.metrics import mean_squared_error
import os
from tqdm import tqdm
import sys
import argparse

from nctc_detect import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
# ***** This file contains the implementations of four CTC algorithms ***** #

# **** This file contains the experimental configuration **** #

def nctc_flow_build(nctc_interval_dir, test_csv_dir, save_root):
	for nctc_file in os.listdir(nctc_interval_dir):
		if nctc_file.endswith('.csv') and (not nctc_file.endswith('http_intervals.csv')):

			save_dir = os.path.join(save_root, nctc_file.replace('.csv', '_flow'))
			os.makedirs(save_dir, exist_ok=True)
			print("Save_dir:", save_dir)

			# English note retained from the original workflow.
			copy_directory(test_csv_dir, save_dir)

			interval_path = os.path.join(nctc_interval_dir, nctc_file)
			df = pd.read_csv(interval_path, header=0)
			intervals = df['IPDs'].tolist()

			total_len = 0
			for root, dirs, files in os.walk(save_dir):
				for file in files:
					if file.endswith('.csv'):
						file_path = os.path.join(root, file)
						df = pd.read_csv(file_path, header=0)
						interval_replace = intervals[total_len: total_len + len(df)]
						total_len += len(df)

						if 'interval' not in df.columns:
							raise ValueError(f"'interval' column not found in {file_path}")

						df['interval'] = interval_replace
						df.to_csv(file_path, index=False)



# -------------------- LNCTC ------------------------- #
class IpdsGeneratorLNCTC():
	def __init__(self, parameters: dict):
		self._parameters = parameters

	def _msgs_generation(self, msgs_len: int):
		msgs = ''
		for i in range(msgs_len):
			msgs += str(random.randint(0, 1))
		return msgs

	def _possible_comb_generation(self, n: int, K: int):
		if n == 1:
			return [(i,) for i in range(K + 1)]
		else:
			res = list()
			for i in range(K + 1):
				for subcomb in self._possible_comb_generation(n - 1, K - i):
					res.append((i,) + subcomb)
			return res

	def _encode_msg(self, msg: str, combs: list):
		comb = combs[int(msg, 2)]
		ipds = [self._parameters["big_delta"] + i * self._parameters["small_delta"] for i in comb]
		return ipds

	def generate_LNCTC_ipds(self):
		msg_length = int((self._parameters["ipds_count"] / self._parameters["n"]) * self._parameters["L"])
		msgs = self._msgs_generation(msg_length)
		combs = self._possible_comb_generation(self._parameters["n"], self._parameters["K"])
		LNCTC_ipds = list()
		for msg_group_index in range(int(len(msgs) / self._parameters["L"])):
			msg = msgs[msg_group_index * self._parameters["L"]: (msg_group_index + 1) * self._parameters["L"]]
			LNCTC_ipds += self._encode_msg(msg, combs)
		pd.DataFrame({"IPDs": LNCTC_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)

def recursive_lnctc_overwrite_and_detect(reference_dir, test_csv_dir, save_dir, max_attempts=2000):
	# English note retained from the original workflow.
	device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
	print(f"[INFO] Using device: {device}")
	TRANSFORMER_PARAMETERS = {
		'discretized_size': 16,
		'feature_dim': 16,
		'd_model': 100,
		'nhead': 4,
		'num_layers': 2,
		'dim_feedforward': 256,
		'dropout': 0.1,
		'is_middle': True
	}
	model_registry = {
		"Transformer_packets_feature_model": (Transformer_packets_feature_model, transformer_packet_detect, TRANSFORMER_PARAMETERS),
	}
	ModelCls, detect_fn, default_params = model_registry["Transformer_packets_feature_model"]
	model = ModelCls(**default_params).to(device)
	checkpoint_path = "./checkpoint_dir/epoch.pth"
	model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	model.eval()
	criterion = nn.CrossEntropyLoss()

	# English note retained from the original workflow.
	normal_scores = scores_calculate(
		'Transformer_packets_feature_model', 'reference',
		model, criterion, reference_dir, 16, 8, 1.389503, 0, 128, True
	)

	# English note retained from the original workflow.
	for root, _, files in os.walk(test_csv_dir):
		# English note retained from the original workflow.
		rel_dir = os.path.relpath(root, test_csv_dir)
		if rel_dir == ".":
			# English note retained from the original workflow.
			rel_dir = ""  # English note retained from the original workflow.

		# English note retained from the original workflow.
		csv_files = [f for f in files if f.lower().endswith(".csv")]
		if not csv_files:
			continue

		# English note retained from the original workflow.
		save_subdir = os.path.join(save_dir, rel_dir) if rel_dir != "" else save_dir
		os.makedirs(save_subdir, exist_ok=True)

		saved_ok = 0  # English note retained from the original workflow.
		desc_text = rel_dir if rel_dir != "" else "(root)"

		with tqdm(total=len(csv_files), desc=desc_text, unit="file") as pbar:
			pbar.set_postfix(saved=saved_ok)

			for file in csv_files:
				highest_score = 0.0
				full_path = os.path.join(root, file)
				try:
					df = pd.read_csv(full_path)
				except Exception as e:
					print(f"[READ ERROR] {full_path}: {e}")
					pbar.update(1)
					pbar.set_postfix(saved=saved_ok)
					continue

				if 'interval' not in df.columns:
					print(f"[ERROR] 'interval' column not found in {full_path}")
					pbar.update(1)
					pbar.set_postfix(saved=saved_ok)
					continue

				row_count = len(df)

				# English note retained from the original workflow.
				success = False
				for attempt in range(max_attempts):
					# English note retained from the original workflow.
					random_seed = random.randint(0, 999999)
					random.seed(random_seed)
					np.random.seed(random_seed)

					# English note retained from the original workflow.
					temp_path = os.path.join(save_dir, "_temp.csv")  # English note retained from the original workflow.
					parameters = {
						"n": 3,
						"K": 13,
						"L": 8,
						"ipds_count": row_count + 3,  # English note retained from the original workflow.
						"big_delta": 20,
						"small_delta": 10,
						"save_generated_ipds_path": temp_path
					}
					gen = IpdsGeneratorLNCTC(parameters)
					gen.generate_LNCTC_ipds()

					try:
						ipds = pd.read_csv(temp_path)["IPDs"].tolist()
					except Exception as e:
						print(f"[TEMP READ ERROR] {temp_path}: {e}")
						break

					# English note retained from the original workflow.
					if len(ipds) < row_count:
						print(f"[ERROR] Generated IPDs length ({len(ipds)}) < row count ({row_count})")
						continue
					elif len(ipds) > row_count:
						ipds = ipds[:row_count]

					# English note retained from the original workflow.
					df_mod = df.copy()
					df_mod["interval"] = ipds

					# English note retained from the original workflow.
					try:
						df_mod.to_csv(temp_path, index=False)
					except Exception as e:
						print(f"[TEMP WRITE ERROR] {temp_path}: {e}")
						break

					# English note retained from the original workflow.
					score = scores_calculate(
						'Transformer_packets_feature_model', 'detect',
						model, criterion, temp_path, 16, 8, 1.389503, 0, 128, True
					)
					score = score[0]

					count_less = sum(1 for x in normal_scores if x <= score)
					score_ratio = count_less / len(normal_scores)
					# print(f"[{desc_text}] {file} | Attempt {attempt+1} | Score: {score:.6f} | Ratio: {score_ratio:.4f}")

					# English note retained from the original workflow.
					if score_ratio > 0.97:
						save_path = os.path.join(save_subdir, file)  # English note retained from the original workflow.
						try:
							df_mod.to_csv(save_path, index=False)
							print(f"[✓] Saved: {save_path}")
							saved_ok += 1
							success = True
						except Exception as e:
							print(f"[SAVE ERROR] {save_path}: {e}")
						break
					elif score_ratio > highest_score:
						highest_score = score_ratio
						best_df_mod = df_mod.copy()

				if not success and best_df_mod is not None:
					save_path = os.path.join(save_subdir, file)
					best_df_mod.to_csv(save_path, index=False)
					print('highest_score = ', highest_score)
					saved_ok += 1

				# English note retained from the original workflow.
				pbar.update(1)  # English note retained from the original workflow.
				pbar.set_postfix(saved=saved_ok)  # English note retained from the original workflow.


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Run LNCTC generation and detection")
	parser.add_argument("--subdir", type=str, default="", help="Name of the subdirectory to process")
	args = parser.parse_args()
	sub_dir = args.subdir

	reference_dir = './normal_dir'
	test_csv_dir = './test_dir'
	save_dir = './lnctc_dir'
	test_csv_dir = os.path.join(test_csv_dir, sub_dir)
	save_dir = os.path.join(save_dir, sub_dir)
	recursive_lnctc_overwrite_and_detect(reference_dir, test_csv_dir, save_dir)







	
