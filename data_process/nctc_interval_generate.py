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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
# ***** This file contains the implementations of four CTC algorithms ***** #

# **** This file contains the experimental configuration **** #

def nctc_flow_build(nctc_interval_dir, test_csv_dir, save_root):
	for nctc_file in os.listdir(nctc_interval_dir):
		if nctc_file.endswith('.csv') and (not nctc_file.endswith('normal_intervals.csv')):

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

# ----------------------- CTCs ---------------------------- #
IPDS_COUNT = 23000000
# IPDS_COUNT = 10
REFERENCE_DATA_DIRPATH = './reference_dir/'
IPDS_SAVE_DIRPATH = './nctc_interval_dir/'

# IPCTC
# IPCTC_PARAMETERS = {
# 	"time_interval": [20, 40, 60],
# 	"rotated_interval": 100,
# 	"noise": None,
# 	"frame_size": 100,
# 	"ipds_count": IPDS_COUNT,
# 	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "IPCTC_1.csv"
# }

IPCTC_PARAMETERS = {
	"time_interval": [10, 20, 30],
	"rotated_interval": 100,
	"noise": None,
	"frame_size": 100,
	"ipds_count": IPDS_COUNT,
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "IPCTC_2.csv"
}

# TRCTC
# TRCTC_PARAMETERS = {
# 	"ipds_count": IPDS_COUNT,
# 	"legit_dataset_path": REFERENCE_DATA_DIRPATH + "normal_intervals.csv",
# 	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "TRCTC_1.csv",
# 	"rules": {'0':[(0, 0.40)], '1':[(0.60, 1)]},
# }
TRCTC_PARAMETERS = {
	"ipds_count": IPDS_COUNT,
	"legit_dataset_path": REFERENCE_DATA_DIRPATH + "normal_intervals.csv",
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "TRCTC_2.csv",
	"rules": {'0':[(0, 0.45)], '1':[(0.55, 1)]},
}

# Jitterbug
# Jitterbug_PARAMETERS = {
# 	"ipds_count": IPDS_COUNT,
# 	"legit_dataset_path": REFERENCE_DATA_DIRPATH + "normal_intervals.csv",
# 	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "Jitterbug_1.csv",
# 	"omega": 20, #ms
# }
Jitterbug_PARAMETERS = {
	"ipds_count": IPDS_COUNT,
	"legit_dataset_path": REFERENCE_DATA_DIRPATH + "normal_intervals.csv",
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "Jitterbug_2.csv",
	"omega": 5, #ms
}



# LNCTC
# LNCTC_PARAMETERS = {
# 	"ipds_count": IPDS_COUNT,
# 	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "LNCTC_1.csv",
# 	"big_delta": 20,
# 	"small_delta": 10,
# 	"L": 8,
# 	"n": 3,
# 	"K": 13
# }

LNCTC_PARAMETERS = {
	"ipds_count": IPDS_COUNT,
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "LNCTC_2.csv",
	"big_delta": 10,
	"small_delta": 5,
	"L": 8,
	"n": 3,
	"K": 13
}
# KLIBUR
KLIBUR_PARAMETERS_1 = {
	"ipds_count": IPDS_COUNT,
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "KLIBUR_1.csv",
	"tau": 5  # English note retained from the original workflow.

}
KLIBUR_PARAMETERS_2= {
	"ipds_count": IPDS_COUNT,
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "KLIBUR_2.csv",
	"tau": 10  # English note retained from the original workflow.

}

KLIBUR_O_PARAMETERS_1= {
	"ipds_count": IPDS_COUNT,
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "KLIBUR_O_1.csv",
	"tau": 5,  # English note retained from the original workflow.
	"outlier_prob": 0.05,
	"outlier_multiplier": 10.0 
}

KLIBUR_O_PARAMETERS_2= {
	"ipds_count": IPDS_COUNT,
	"save_generated_ipds_path": IPDS_SAVE_DIRPATH + "KLIBUR_O_2.csv",
	"tau": 5,  # English note retained from the original workflow.
	"outlier_prob": 0.1,
	"outlier_multiplier": 10.0 
}

# ----------------------- KLIBUR ---------------------------- #
class IpdsGeneratorKLIBUR():
	def __init__(self, parameters: dict):
		self._parameters = parameters

	def _generate_covert_msgs(self, length: int):
		msgs = ''
		for i in range(length):
			msgs += str(random.randint(0, 1))
		return msgs

	def _generate_ipds(self, msgs: str, tau: int):
		ipd_len = int(len(msgs))
		ipds = list()
		threshold = 1.5 * tau
		candidates = np.arange(threshold, 2.4 * tau, 0.001)
		for msg in msgs:
			if msg == '1':
				ipd = tau
				ipds.append(ipd)
			else:
				ipd = tau * 2
				ipds.append(ipd)
		
		for i, ipd in enumerate(tqdm(ipds, desc="Generating KLIBUR IPDs...")):
			if ipd <= threshold:
				new_ipd = abs(np.random.normal(loc=0, scale=threshold/7.0))
				if new_ipd > threshold:
					new_ipd = threshold
				ipds[i] = new_ipd
			else:
				# English note retained from the original workflow.
				ipds[i] = np.random.choice(candidates)
		return ipds


	def generate_KLIBUR_ipds(self):
		# parameters: dict-like object containing time_interval, rotated_interval, noise, frame_size and msg_length
		msgs = self._generate_covert_msgs(self._parameters["ipds_count"])
		ipds = self._generate_ipds(msgs, self._parameters["tau"])
		pd.DataFrame({"IPDs": ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)

# ----------------------- KLIBUR_O ---------------------------- #
class IpdsGeneratorKLIBUR_O():
	"""
	-κlibur-O:
	  English documentation retained from the original workflow.
	  English documentation retained from the original workflow.
	"""
	def __init__(self, parameters: dict):
		self._parameters = parameters
	def _generate_covert_msgs(self, length: int):
		return ''.join(str(random.randint(0, 1)) for _ in range(length))

	def _generate_ipds(self, msgs: str, tau: float):
		ipds = []
		threshold = 1.5 * tau
		# English note retained from the original workflow.
		candidates = np.arange(threshold, 2.4 * tau, 0.001)

		# English note retained from the original workflow.
		for ch in msgs:
			ipds.append(tau if ch == '1' else 2 * tau)

		# English note retained from the original workflow.
		p = float(self._parameters["outlier_prob"])
		k = float(self._parameters["outlier_multiplier"])

		for i, ipd in enumerate(tqdm(ipds, desc="Generating KLIBUR-O IPDs...")):
			if ipd <= threshold:
				# English note retained from the original workflow.
				new_ipd = abs(np.random.normal(loc=0.0, scale=threshold / 7.0))
				if new_ipd > threshold:
					new_ipd = threshold
				ipds[i] = new_ipd
			else:
				# English note retained from the original workflow.
				if np.random.rand() < p:
					ipds[i] = k * tau
				else:
					ipds[i] = np.random.choice(candidates)
		return ipds

	def generate_KLIBUR_O_ipds(self):
		msgs = self._generate_covert_msgs(self._parameters["ipds_count"])
		ipds = self._generate_ipds(msgs, float(self._parameters["tau"]))
		pd.DataFrame({"IPDs": ipds}).to_csv(self._parameters["save_generated_ipds_path"], index=False)

# ----------------------- IPCTC ---------------------------- #
class IpdsGeneratorIPCTC():
	def __init__(self, parameters: dict):
		self._parameters = parameters

	def _generate_covert_msgs(self, length: int):
		msgs = ''
		for i in range(length):
			msgs += str(random.randint(0, 1))
		return msgs

	def _generate_ipds(self, msgs: str, time_interval: list, rotated_interval: int, frame_size: int, noise: int):
		frame_count = int(len(msgs) / frame_size) if len(msgs) % frame_size == 0 else int(len(msgs) / frame_size) + 1
		frames = list()
		msg_count = 0
		interval_num = 0
		for i in tqdm(range(frame_count), desc="Generating IPCTC IPDs..."):
			ipds = list()
			ipd = time_interval[interval_num % len(time_interval)] / 2
			msgs_frame = msgs[i * frame_size: -1 if i == frame_count - 1 else (i + 1) * frame_size]
			for msg in msgs_frame:
				if msg == '1':
					ipds.append(ipd)
					ipd = time_interval[interval_num % len(time_interval)]
				else:
					ipd += time_interval[interval_num % len(time_interval)]
				msg_count += 1
				if msg_count == rotated_interval:
					interval_num += 1
			frames.append(ipds)
		return frames

	def _merge_frame(self, frames: list):
		res = list()
		for frame in frames:
			res += frame
		return res

	def generate_IPCTC_ipds(self):
		# parameters: dict-like object containing time_interval, rotated_interval, noise, frame_size and msg_length
		msgs = self._generate_covert_msgs(4 * self._parameters["ipds_count"])
		frames = self._generate_ipds(msgs, self._parameters["time_interval"], self._parameters["rotated_interval"], self._parameters["frame_size"], self._parameters["noise"])
		total_ipds = self._merge_frame(frames)
		pd.DataFrame({"IPDs": total_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)


# ----------------------- TRCTC----------------------------- #
class IpdsGeneratorTRCTC():
	def __init__(self, parameters: dict):
		self._parameters = parameters

	def _msgs_generation(self, msgs_len: int):
		msgs = ''
		for i in range(msgs_len):
			msgs += str(random.randint(0, 1))
		return msgs

	def _ipds_load(self, ipd_filepath: str, fillna_method: str = "ffill", multiply: int = None):
		ipds = pd.read_csv(ipd_filepath)["IPDs"]
		if fillna_method != None:
			ipds = ipds.fillna(method = fillna_method)
		if multiply != None:
			ipds = ipds * multiply
		ipds = ipds[ipds[ipds > 0].index].reset_index(drop = True)
		return ipds

	def _generate_ipds(self, msgs: str, rules: list, ipds: pd.Series):
		traffic_ipds = list()
		sorted_ipds = sorted(ipds)
		for msg in tqdm(msgs, desc="Generating TRCTC IPDs..."):
			random_bin = rules[msg][random.randint(0, len(rules[msg]) - 1)]
			traffic_ipds.append(sorted_ipds[random.randint(int(len(sorted_ipds) * random_bin[0]), int(len(sorted_ipds) * random_bin[1]) - 1)])
		return traffic_ipds

	def generate_TRCTC_ipds(self):
		msgs = self._msgs_generation(self._parameters["ipds_count"])
		legit_idps = self._ipds_load(self._parameters["legit_dataset_path"])
		TRCTC_ipds = self._generate_ipds(msgs, self._parameters["rules"], legit_idps)
		pd.DataFrame({"IPDs": TRCTC_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)


# ----------------------- Jitterbug ------------------------ #
class IpdsGeneratorJitterbug():
	def __init__(self, parameters: dict):
		self._parameters = parameters

	def _msgs_generation(self, msgs_len: int):
		msgs = ''
		for i in range(msgs_len):
			msgs += str(random.randint(0, 1))
		return msgs

	def _ipds_load(self, ipd_filepath: str, fillna_method: str = "ffill", multiply: int = None):
		ipds = pd.read_csv(ipd_filepath)["IPDs"]
		if fillna_method != None:
			ipds = ipds.fillna(method = fillna_method)
		if multiply != None:
			ipds = ipds * multiply
		ipds = ipds[ipds[ipds > 0].index].reset_index(drop = True)
		return list(ipds[0: self._parameters["ipds_count"]].values)

	def _generate_ipds(self, msgs: str, omega: float, ipds: list):
		if len(msgs) > len(ipds):
			times = math.ceil(len(msgs) / len(ipds))
			ipds = (ipds * times)[:len(msgs)]
		for i in tqdm(range(len(msgs)), desc="Generating Jitterbug IPDs..."):
			if msgs[i] == '0':
				ipds[i] += (omega - ipds[i] % omega) if ipds[i] % omega != 0 else 0
			else:
				ipds[i] += (omega / 2 - ipds[i] % omega) if ipds[i] % omega <= omega / 2 else (3 * omega / 2 - ipds[i] % omega)
		return ipds

	def generate_Jitterbug_ipds(self):
		msgs = self._msgs_generation(self._parameters["ipds_count"])
		legit_idps = self._ipds_load(self._parameters["legit_dataset_path"])
		Jitterbug_ipds = self._generate_ipds(msgs, self._parameters["omega"], legit_idps)
		pd.DataFrame({"IPDs": Jitterbug_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)

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
		for msg_group_index in tqdm(range(int(len(msgs) / self._parameters["L"])), desc="Generating LNCTC IPDs..."):
			msg = msgs[msg_group_index * self._parameters["L"]: (msg_group_index + 1) * self._parameters["L"]]
			LNCTC_ipds += self._encode_msg(msg, combs)
		pd.DataFrame({"IPDs": LNCTC_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)


def collect_normal_intervals(test_dir, save_dir):
	ipd_lists = []
	for root, _, files in os.walk(test_dir):
		for file in files:
			if file.endswith('.csv'):
				file_path = os.path.join(root, file)
				try:
					df = pd.read_csv(file_path, header=0)
				except pd.errors.EmptyDataError:
					print(f"Warning: {file_path} is empty and will be skipped.")
					continue
				ipd_lists.append(df['interval'].tolist())

	os.makedirs(save_dir, exist_ok=True)
	normal_save_path = os.path.join(save_dir, 'normal_intervals.csv')
	merged = []
	for ipd_list in ipd_lists:
		merged.extend(ipd_list)
	pd.DataFrame({'IPDs': merged}).to_csv(normal_save_path, index=False)
	print(f"Saved normal intervals to {normal_save_path}")


def update_paths_and_count(reference_dir, save_dir, ipds_count):
	global REFERENCE_DATA_DIRPATH, IPDS_SAVE_DIRPATH, IPDS_COUNT
	REFERENCE_DATA_DIRPATH = reference_dir if reference_dir.endswith(os.sep) else reference_dir + os.sep
	IPDS_SAVE_DIRPATH = save_dir if save_dir.endswith(os.sep) else save_dir + os.sep
	IPDS_COUNT = ipds_count
	os.makedirs(IPDS_SAVE_DIRPATH, exist_ok=True)

	for params in [
		IPCTC_PARAMETERS, TRCTC_PARAMETERS, Jitterbug_PARAMETERS, LNCTC_PARAMETERS,
		KLIBUR_PARAMETERS_1, KLIBUR_PARAMETERS_2, KLIBUR_O_PARAMETERS_1, KLIBUR_O_PARAMETERS_2,
	]:
		params["ipds_count"] = IPDS_COUNT

	IPCTC_PARAMETERS["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "IPCTC.csv")
	TRCTC_PARAMETERS["legit_dataset_path"] = os.path.join(REFERENCE_DATA_DIRPATH, "normal_intervals.csv")
	TRCTC_PARAMETERS["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "TRCTC.csv")
	Jitterbug_PARAMETERS["legit_dataset_path"] = os.path.join(REFERENCE_DATA_DIRPATH, "normal_intervals.csv")
	Jitterbug_PARAMETERS["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "Jitterbug.csv")
	LNCTC_PARAMETERS["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "LNCTC.csv")
	KLIBUR_PARAMETERS_1["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "KLIBUR_tau5.csv")
	KLIBUR_PARAMETERS_2["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "KLIBUR_tau10.csv")
	KLIBUR_O_PARAMETERS_1["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "KLIBUR_O_tau5.csv")
	KLIBUR_O_PARAMETERS_2["save_generated_ipds_path"] = os.path.join(IPDS_SAVE_DIRPATH, "KLIBUR_O_tau10.csv")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate synthetic NCTC IPDs and optionally write them into test flows")
	parser.add_argument("--method", choices=["ipctc", "trctc", "jitterbug", "lnctc", "klibur", "klibur_o", "all"], default="all")
	parser.add_argument("--reference_dir", default="./reference_dir")
	parser.add_argument("--save_dir", default="./nctc_interval_dir")
	parser.add_argument("--test_dir", default="./test_dir")
	parser.add_argument("--ipds_count", type=int, default=IPDS_COUNT)
	parser.add_argument("--collect_reference", action="store_true")
	parser.add_argument("--build_flow", action="store_true")
	args = parser.parse_args()

	update_paths_and_count(args.reference_dir, args.save_dir, args.ipds_count)
	if args.collect_reference:
		collect_normal_intervals(args.test_dir, args.reference_dir)

	if args.method in ("ipctc", "all"):
		IpdsGeneratorIPCTC(IPCTC_PARAMETERS).generate_IPCTC_ipds()
	if args.method in ("trctc", "all"):
		IpdsGeneratorTRCTC(TRCTC_PARAMETERS).generate_TRCTC_ipds()
	if args.method in ("jitterbug", "all"):
		IpdsGeneratorJitterbug(Jitterbug_PARAMETERS).generate_Jitterbug_ipds()
	if args.method in ("lnctc", "all"):
		IpdsGeneratorLNCTC(LNCTC_PARAMETERS).generate_LNCTC_ipds()
	if args.method in ("klibur", "all"):
		IpdsGeneratorKLIBUR(KLIBUR_PARAMETERS_1).generate_KLIBUR_ipds()
		IpdsGeneratorKLIBUR(KLIBUR_PARAMETERS_2).generate_KLIBUR_ipds()
	if args.method in ("klibur_o", "all"):
		IpdsGeneratorKLIBUR_O(KLIBUR_O_PARAMETERS_1).generate_KLIBUR_O_ipds()
		IpdsGeneratorKLIBUR_O(KLIBUR_O_PARAMETERS_2).generate_KLIBUR_O_ipds()

	if args.build_flow:
		nctc_flow_build(nctc_interval_dir=IPDS_SAVE_DIRPATH, test_csv_dir=args.test_dir, save_root=IPDS_SAVE_DIRPATH)
	print('finished')




	
