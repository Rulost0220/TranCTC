from scapy.all import RawPcapReader, Ether
import os
from scapy.layers.inet6 import IPv6, TCP
import numpy as np
import pandas as pd
import hashlib
import csv
from collections import defaultdict
from tqdm import tqdm
from itertools import tee, islice
from pathlib import Path
import ipaddress
import shutil
import random
import shutil
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import os
def _read_numeric_series_from_txt(txt_path: str) -> np.ndarray:

    # English note retained from the original workflow.
    try:
        df = pd.read_csv(txt_path, header=None, sep=None, engine="python")
        vals = pd.to_numeric(df.values.ravel(order="C"), errors="coerce")
        vals = pd.Series(vals).dropna().to_numpy()
        if vals.size > 0:
            return vals
    except Exception:
        pass

    # English note retained from the original workflow.
    num_pat = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    buf = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for m in num_pat.findall(line):
                try:
                    buf.append(float(m))
                except Exception:
                    continue
    return np.asarray(buf, dtype=float)


def _ipd_value_generator(ipd_txt_dir: str):

    if not os.path.isdir(ipd_txt_dir):
        raise FileNotFoundError(f"ipd_txt_dir not found: {ipd_txt_dir}")

    txt_files = [os.path.join(ipd_txt_dir, f)
                 for f in os.listdir(ipd_txt_dir)
                 if f.lower().endswith(".txt")]
    txt_files.sort()  # English note retained from the original workflow.

    if not txt_files:
        raise ValueError(f"No .txt files found under {ipd_txt_dir}")

    total_count = 0
    for path in txt_files:
        arr = _read_numeric_series_from_txt(path)
        if arr.size == 0:
            continue
        arr = arr * 1000.0
        total_count += arr.size
        for v in arr:
            yield v

    if total_count == 0:
        raise ValueError(f"No numeric IPD values found in any txt under {ipd_txt_dir}")


def fill_ipd_dir_into_csvs_multi(ipd_txt_dir: str, test_csv_dir: str, save_dirs: list) -> int:

    if not os.path.isdir(test_csv_dir):
        raise FileNotFoundError(f"test_csv_dir not found: {test_csv_dir}")

    if not isinstance(save_dirs, (list, tuple)):
        raise ValueError("save_dirs must be a list of directories")

    # English note retained from the original workflow.
    ipd_gen = _ipd_value_generator(ipd_txt_dir)

    processed = 0
    for root, _, files in os.walk(test_csv_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".csv"):
                continue

            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, test_csv_dir)

            # English note retained from the original workflow.
            df = pd.read_csv(src_path, header=0)
            if "interval" not in df.columns:
                raise ValueError(f"'interval' column not found in {src_path}")

            n = len(df)
            if n == 0:
                # English note retained from the original workflow.
                for sd in save_dirs:
                    dst_path = os.path.join(sd, rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    df.to_csv(dst_path, index=False)
                processed += 1
                continue

            # English note retained from the original workflow.
            take = []
            try:
                for _ in range(n):
                    take.append(next(ipd_gen))
            except StopIteration:
                raise ValueError(
                    f"Not enough IPD values across all txt files in {ipd_txt_dir} "
                    f"to fill {src_path} (need {n}, got {len(take)} so far)."
                )

            df["interval"] = np.asarray(take, dtype=float)

            # English note retained from the original workflow.
            for sd in save_dirs:
                dst_path = os.path.join(sd, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                df.to_csv(dst_path, index=False)

            processed += 1

    return processed

def fill_ipd_into_csvs(ipd_csv_path: str, test_csv_dir: str, save_dir: str) -> int:

    # English note retained from the original workflow.
    ipd_df = pd.read_csv(ipd_csv_path, header=None)
    # English note retained from the original workflow.
    if len(ipd_df) > 0 and str(ipd_df.iat[0, 0]).strip().upper() == "IPDS":
        ipd_series = pd.to_numeric(ipd_df.iloc[1:, 0], errors="coerce").dropna().reset_index(drop=True)
    else:
        ipd_series = pd.to_numeric(ipd_df.iloc[:, 0], errors="coerce").dropna().reset_index(drop=True)

    if len(ipd_series) == 0:
        raise ValueError(f"No numeric IPD values found in {ipd_csv_path}")

    ipd_vals = ipd_series.to_numpy()
    ipd_len = len(ipd_vals)

    # English note retained from the original workflow.
    total_idx = 0  # English note retained from the original workflow.
    processed = 0

    for root, _, files in os.walk(test_csv_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".csv"):
                continue
            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, test_csv_dir)
            dst_path = os.path.join(save_dir, rel_path)

            # English note retained from the original workflow.
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # English note retained from the original workflow.
            df = pd.read_csv(src_path, header=0)
            if 'interval' not in df.columns:
                raise ValueError(f"'interval' column not found in {src_path}")

            n = len(df)
            if n <= 0:
                # English note retained from the original workflow.
                df.to_csv(dst_path, index=False)
                continue

            # English note retained from the original workflow.
            # English note retained from the original workflow.
            take_idx = np.arange(total_idx, total_idx + n) % ipd_len
            interval_replace = ipd_vals[take_idx]
            total_idx = (total_idx + n) % ipd_len  # English note retained from the original workflow.

            # English note retained from the original workflow.
            df['interval'] = interval_replace
            df.to_csv(dst_path, index=False)
            processed += 1

    return processed

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Write generated IPD sequences into flow CSV files")
	parser.add_argument("--mode", choices=["single", "multi", "flow"], default="flow")
	parser.add_argument("--ipd_csv", type=str, default="./ipd_dir/ipd.csv")
	parser.add_argument("--ipd_txt_dir", type=str, default="./ipd_dir")
	parser.add_argument("--nctc_interval_dir", type=str, default="./nctc_interval_dir")
	parser.add_argument("--test_csv_dir", type=str, default="./test_dir")
	parser.add_argument("--save_dir", type=str, default="./nctc_dir")
	parser.add_argument("--save_dirs", nargs="*", default=["./nctc_dir"])
	args = parser.parse_args()

	if args.mode == "single":
		processed_count = fill_ipd_into_csvs(args.ipd_csv, args.test_csv_dir, args.save_dir)
	elif args.mode == "multi":
		processed_count = fill_ipd_dir_into_csvs_multi(args.ipd_txt_dir, args.test_csv_dir, args.save_dirs)
	else:
		nctc_flow_build(args.nctc_interval_dir, args.test_csv_dir, args.save_dir)
		processed_count = "flow mode finished"

	print("Processed CSV count:", processed_count)
