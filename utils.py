import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import csv
import os
import re
from tqdm import tqdm
from scapy.all import rdpcap, IPv6, TCP, UDP, RawPcapReader, Ether, conf, IP
from ipaddress import IPv6Address
import pandas as pd
from scapy.layers.inet6 import IPv6, IPv6ExtHdrHopByHop, IPv6ExtHdrRouting, IPv6ExtHdrFragment, IPv6ExtHdrDestOpt
from scapy.layers.inet import TCP, UDP   
from math import log2
from pathlib import Path
import random
random.seed(42)
import argparse
import shutil
import torch
import torch.nn as nn 
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import random
from typing import List, Optional



def _collect_csv_files(path_or_dir):
    """Supports a single CSV file, a directory, or a file list."""
    if isinstance(path_or_dir, (list, tuple)):
        files = [p for p in path_or_dir if str(p).lower().endswith(".csv") and os.path.isfile(p)]
        return files
    path_or_dir = str(path_or_dir)
    if os.path.isfile(path_or_dir):
        return [path_or_dir] if path_or_dir.lower().endswith(".csv") else []
    if os.path.isdir(path_or_dir):
        out = []
        for root, _, files in os.walk(path_or_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    out.append(os.path.join(root, f))
        return out
    raise ValueError(f"Path does not exist: {path_or_dir}")

def copy_directory(src_dir, dest_dir):
    """
    English documentation retained from the original workflow.
    
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    """
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory {src_dir} does not exist")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # English note retained from the original workflow.
    
    # English note retained from the original workflow.
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)  # English note retained from the original workflow.


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def warmup_lambda(epoch):
    if epoch < 10:
        return epoch / 10
    return 1

def mask_data(data):
    # English note retained from the original workflow.
    data = np.array(data, dtype=int)  # English note retained from the original workflow.

    # English note retained from the original workflow.
    # English note retained from the original workflow.
    # English note retained from the original workflow.
    # English note retained from the original workflow.
    modified_f = data & 0b0101  
    modified_b = data & 0b1010
    return modified_f, modified_b

def construct_single_ipd_prediction_dataset(raw_data: np.ndarray, raw_len: int, time_step_x: int, time_step_y: int, if_unique: bool = False, is_middle: bool = False):
    # This method constructs the sequential-prediction dataset;
    # "time_step_x": IPD picking step (normally = 1)
    # "time_step_y": variation pattern length
    raw_data = np.array(raw_data)
    if raw_len != 0:
        raw_data = raw_data[:raw_len]
    rolling_matrix_rows_count = int(len(raw_data) / time_step_x)
    rolling_matrix = raw_data[:rolling_matrix_rows_count * time_step_x].reshape((rolling_matrix_rows_count, time_step_x)).T
    assert rolling_matrix_rows_count > time_step_y + 1
    temp_dataset_shape = (time_step_x * (rolling_matrix_rows_count - time_step_y), time_step_y + 1)
    temp_dataset = np.zeros(shape = temp_dataset_shape)
    idx = 0
    for row in rolling_matrix:
        for i in range(len(row) - time_step_y):
            temp_dataset[idx] = row[i: i + time_step_y + 1]
            idx += 1

    if is_middle:  # English note retained from the original workflow.
        mid_idx = int((time_step_y // 2))
        dataset_X_f = temp_dataset[:, :mid_idx]
        dataset_X_b = temp_dataset[:, mid_idx + 1:]
        # English note retained from the original workflow.
        
        modified_f, _ = mask_data(dataset_X_f[:, -1])
        _, modified_b = mask_data(dataset_X_b[:, 0])
        dataset_X_f[:, -1] = modified_f
        dataset_X_b[:, 0] = modified_b

        # English note retained from the original workflow.
        dataset_X = np.hstack((dataset_X_f, dataset_X_b))
        dataset_Y = temp_dataset[:, mid_idx]
    else:
        dataset_X = temp_dataset[:, :-1]
        modified_f, modified_b = mask_data(dataset_X[:, -1])
        dataset_X[:, -1] = modified_f
        dataset_Y = temp_dataset[:,-1]
    return dataset_X, dataset_Y


def gas_ipd_discretization(ipd_list, median=None):
    # English note retained from the original workflow.
    if median is None:
        diffs = [abs(ipd_list[i+1] - ipd_list[i]) for i in range(len(ipd_list) - 1)]
        diffs_sorted = sorted(diffs)
        n = len(diffs_sorted)
        if n%2 == 1:
            median_idx = int((n - 1) / 2)
        else:
            median_idx = int((n / 2) - 1)
        median = diffs_sorted[median_idx]
    # print('median = ',median)

    discretization_list = []
    fv,bv,fs,bs = 0,0,0,0
    for i in range(1, len(ipd_list) - 1):
        fv = (1 if abs(ipd_list[i + 1] - ipd_list[i]) >= median else 0)
        bv = (1 if abs(ipd_list[i] - ipd_list[i - 1]) >= median else 0)
        fs = (1 if (ipd_list[i + 1] >= ipd_list[i]) else 0)
        bs = (1 if (ipd_list[i] >= ipd_list[i-1]) else 0)
        discretization = (1.0 * bs) + (2.0 * fs) + (4.0 * bv) + (8.0 * fv)
        discretization_list.append(discretization)

    return discretization_list

def ipd_csv_to_ipd_list(csv_path, seq_len=None):
    df = pd.read_csv(csv_path, header=None, skiprows=1)
    ipd_list = df.iloc[:, 0].tolist()  # English note retained from the original workflow.
    if seq_len:
        ipd_list = ipd_list[:seq_len]
    return ipd_list

def flow_csv_to_np_data(csv_path, seq_len, float_fields, median, num_samples):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.

    Args:
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.

    Returns:
        English documentation retained from the original workflow.
    """

    df = pd.read_csv(csv_path)
# English note retained from the original workflow.
#    
#    rng = np.random.default_rng(42)
#
#    n_drop = int(len(df) * 0.15)
#    drop_idx = rng.choice(df.index[1:], size=n_drop, replace=False)
#    drop_idx = np.sort(drop_idx)
#
#    to_drop = set(drop_idx)
#
#    for i in drop_idx:
#        j = i - 1
#        while j in to_drop:
#            j -= 1
#        df.loc[j, "interval"] += df.loc[i, "interval"]
#
#    df = df.drop(drop_idx).reset_index(drop=True)
#
# English note retained from the original workflow.
    intervals = df['interval'].values
    
    float_data = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        try:
            sample = [float(row[f]) for f in float_fields]
        except:
            continue

        fv = float(abs(intervals[i+1] - intervals[i]) > median)
        bv = float(abs(intervals[i] - intervals[i - 1]) > median)
        fs = float(intervals[i + 1] > intervals[i])
        bs = float(intervals[i] > intervals[i - 1])
        label = int(fv) * 8 + int(bv) * 4 + int(fs) * 2 + int(bs)
        sample.extend([label])

        float_data.append(sample)

    # English note retained from the original workflow.
    arr = np.asarray(float_data, dtype=float)
    npy_list = [arr[i:i + seq_len] for i in range(arr.shape[0] - seq_len + 1)]

    merged = np.stack(npy_list, axis=0)

    # English note retained from the original workflow.
    if num_samples and merged.shape[0] >= num_samples:
        merged = merged[0:num_samples, :, :]
    return merged

def has_tcp_or_udp(pkt):  # English note retained from the original workflow.
    while pkt:
        if pkt.haslayer(TCP) or pkt.haslayer(UDP):
            return True
        pkt = pkt.payload
    return False

def parse_ipv6_packet(pkt_data,k):
    
    """Parse IPv6 packets, supporting Ethernet-encapsulated and raw IPv6 packets"""
    pkt = None
    is_ethernet = False

    pkt = Ether(pkt_data)
    if pkt.haslayer(IPv6):
        # English note retained from the original workflow.
        if pkt.haslayer(TCP) or pkt.haslayer(UDP):
            # English note retained from the original workflow.
            return pkt[IPv6]
    elif pkt.haslayer(IP):
        # English note retained from the original workflow.
        is_ethernet = True
    else:
        is_ethernet = False
    if not is_ethernet:
        if (pkt_data[0] >> 4) == 6:
            pkt = IPv6(pkt_data)
            # English note retained from the original workflow.
            if has_tcp_or_udp(pkt):
                # English note retained from the original workflow.
                return pkt
        # elif (pkt_data[0] >> 4) == 4:
        # English note retained from the original workflow.
    return None

def ipds_count(csv_dir):
    total_num = 0
    for dirpath, dirnames, filenames in os.walk(csv_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                full_path = os.path.join(dirpath, filename)
                with open(full_path, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    total_num += sum(1 for _ in reader)
    print(f"Counted {total_num} IPDs")
    return total_num

def load_multi_features_from_csv(csv_source, float_fields, seq_len, median, num_samples=None):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    """
    seq_len += 1
    csv_files = _collect_csv_files(csv_source)
    
    feature_dim = len(float_fields) + 1

    npy_data = []	
    if not num_samples:
        num_samples = float("inf")
    sum = 0
    csv_file_count = 0

    for csv_file in csv_files:
#		print(f"Processing CSV File:  {csv_file_count+1}/{len(csv_files)}")
        csv_file_count += 1
        # print(f"------------Processing CSV File: {csv_file_count}/{len(csv_files)}-----------")
        if sum < num_samples:
            data = flow_csv_to_np_data(csv_file, seq_len, float_fields, median, num_samples=(num_samples-sum))
            sum += data.shape[0]
            npy_data.append(data)
            # print(f"NPY_DATA Extract: {sum}/{num_samples}")
        else:
            break
    
    npy_data_all = np.concatenate(npy_data, axis=0)
    npy_data_all = npy_data_all[:, :, (-1 * (feature_dim)):]  # English note retained from the original workflow.
    # print(npy_data_all.shape)  
    return npy_data_all

def load_data_from_npy_dir(data_dir, feature_dim, num_samples=None):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory")

    # English note retained from the original workflow.
    files = sorted(data_dir.rglob("*.npy"))
    if not files:
        raise ValueError(f"No .npy files found under {data_dir}")

    # English note retained from the original workflow.
    # if num_samples and num_samples > 0:
    #     files = files[:num_samples]
    random.seed(42)  # English note retained from the original workflow.
    files = random.sample(files, num_samples)

    print(f"Found {len(files)} .npy files recursively")

    data_npy = []
    target_shape = None
    skipped = 0

    for fp in files:
        try:
            arr = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"[skip] failed to read {fp}: {e}")
            skipped += 1
            continue

        if arr.ndim == 3:
            arr = arr[0]

        # English note retained from the original workflow.
        if target_shape is None:
            target_shape = arr.shape
        elif arr.shape != target_shape:
            print(f"[skip] shape mismatch {fp.name}: {arr.shape} != {target_shape}")
            skipped += 1
            continue

        data_npy.append(arr)

    if not data_npy:
        raise RuntimeError("No stackable data, possibly because all reads failed or shapes differ")

    data_npy = np.stack(data_npy, axis=0)
    data_npy = data_npy[:, :, (-1 * (feature_dim)):]  # English note retained from the original workflow.
    print(f"loaded: {len(data_npy)} items, skipped: {skipped} items, shape: {data_npy.shape}")
    return data_npy


def check_tcp_seq(csv_dir):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    :return: None
    """
    for dirpath, dirnames, filenames in os.walk(csv_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                full_path = os.path.join(dirpath, filename)
                df = pd.read_csv(full_path)
                tcp_seq = df['tcp_seq'].values
                if not np.all(np.diff(tcp_seq) >= 0):
                    print(f"❌ File {filename} has non-contiguous TCP sequence numbers")
                else:
                    print(f"✅ File {filename} has contiguous TCP sequence numbers")

# def compute_cce_discretized(sequence, m=5, bins=64):
# 	if len(sequence) < m:
# 		return None
# 	hist, bin_edges = np.histogram(sequence, bins=bins, density=False)
# 	discretized = np.digitize(sequence, bin_edges[:-1])
# 	patterns = [tuple(discretized[i:i+m]) for i in range(len(discretized)-m+1)]
# 	count_m = Counter(patterns)
# 	count_m1 = Counter([pat[:-1] for pat in patterns])
# 	total_m = sum(count_m.values())

# 	cond_entropy = 0.0
# 	for pat in count_m:
# 		p_xy = count_m[pat] / total_m
# 		p_x = count_m1[pat[:-1]] / total_m
# 		cond_entropy -= p_xy * np.log2(p_xy / (p_x + 1e-10) + 1e-10)

# 	correction = (bins ** m) / (2 * total_m * np.log(2))
# 	return cond_entropy + correction
def compute_cce_discretized(sequence, m=5, bin_edges=None):
    if len(sequence) < m:
        return None
    
    discretized = np.digitize(sequence, bin_edges, right=False) - 1
    patterns = [tuple(discretized[i:i+m]) for i in range(len(discretized)-m+1)]
    count_m = Counter(patterns)
    count_m1 = Counter([pat[:-1] for pat in patterns])
    total_m = sum(count_m.values())

    cond_entropy = 0.0
    for pat in count_m:
        p_xy = count_m[pat] / total_m
        p_x = count_m1[pat[:-1]] / total_m
        cond_entropy -= p_xy * np.log2(p_xy / (p_x + 1e-10) + 1e-10)

    # Optional: bins ** (m-1) correction term (some variants use this)
    correction = (len(bin_edges) ** m) / (2 * total_m * np.log(2))
    return cond_entropy + correction

def ipd_to_string(ipd_sequence):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    """
    string_sequence = []
    for ipd in ipd_sequence:
        rounded = round(ipd, 4)
        str_ipd = f"{rounded:.4f}".split('.')[1]  # English note retained from the original workflow.
        leading_zeros = len(str_ipd) - len(str_ipd.lstrip('0'))
        letter = chr(ord('A') + leading_zeros)  # A, B, C, ...
        digits = str_ipd.lstrip('0')[:2].ljust(2, '0')  # English note retained from the original workflow.
        string_sequence.append(f"{letter}{digits}")
    return ''.join(string_sequence)


def split_ipds_csv(csv_path: str,
                   save_dir: str,
                   seg_len: int,
                   n: int,
                   *,
                   mode: str = "sequential",
                   prefix: str = "segment",
                   zero_pad: int = 5,
                   include_header: bool = True,
                   step_len: int = 100,
                   max_len_cap: int = 2000,
                   start_offset: int = 0) -> List[str]:
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
                  English documentation retained from the original workflow.
    English documentation retained from the original workflow.

    English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        mode   : 'sequential' | 'reference' | 'nctc'
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.

    English documentation retained from the original workflow.
    """
    if seg_len <= 0 or n <= 0 or step_len <= 0 or max_len_cap <= 0:
        raise ValueError("seg_len, n, step_len, and max_len_cap must be positive integers.")
    if start_offset < 0:
        raise ValueError("start_offset must be >= 0.")
    mode = mode.lower()
    if mode not in {"sequential", "reference", "nctc"}:
        raise ValueError("mode only supports 'sequential', 'reference', or 'nctc'.")

    df = pd.read_csv(csv_path)
    if "IPDs" not in df.columns:
        raise ValueError(f"Input file is missing the 'IPDs' column: {csv_path}")
    vals = df["IPDs"].to_numpy()
    total = len(vals)
    if total <= start_offset:
        raise ValueError(f"Insufficient data: total rows {total} <= start_offset {start_offset}.")

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    ptr = start_offset  # English note retained from the original workflow.

    def save_chunk(arr: np.ndarray, outpath: Path):
        pd.DataFrame({"IPDs": arr}).to_csv(outpath, index=False, header=include_header)
        saved.append(str(outpath.resolve()))

    if mode == "sequential":
        # English note retained from the original workflow.
        for i in range(n):
            end = ptr + seg_len
            if end > total:
                break
            fname = f"{prefix}_{str(i).zfill(zero_pad)}.csv"
            save_chunk(vals[ptr:end], out_dir / fname)
            ptr = end  # English note retained from the original workflow.

    else:
        # English note retained from the original workflow.
        length_grid = []
        L = seg_len
        while True:
            length_grid.append(min(L, max_len_cap))
            if L >= max_len_cap:
                break
            L += step_len

        # English note retained from the original workflow.
        # English note retained from the original workflow.
        counters = {Lk: 0 for Lk in length_grid}  # English note retained from the original workflow.
        for Lk in length_grid:
            for _ in range(n):
                end = ptr + Lk
                if end > total:
                    # English note retained from the original workflow.
                    break
                if mode == "reference":
                    fname = f"{prefix}_len{Lk}_{str(counters[Lk]).zfill(zero_pad)}.csv"
                    outpath = out_dir / fname
                else:  # nctc
                    subdir = out_dir / f"seg_{Lk}"
                    subdir.mkdir(parents=True, exist_ok=True)
                    fname = f"{prefix}_{str(counters[Lk]).zfill(zero_pad)}.csv"
                    outpath = subdir / fname

                save_chunk(vals[ptr:end], outpath)
                counters[Lk] += 1
                ptr = end  # English note retained from the original workflow.

            # English note retained from the original workflow.
            if ptr + min(length_grid[-1], max_len_cap) > total:
                # English note retained from the original workflow.
                pass

    return saved

def flow_csv_count(csv_path, seq_len, float_fields, median):
    """
    Count number of sliding-window samples produced by flow_csv_to_np_data.
    This is EXACTLY equivalent to merged.shape[0].
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)

    intervals = df['interval'].values
    valid_rows = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        try:
            _ = [float(row[f]) for f in float_fields]
        except:
            continue
        valid_rows.append(i)

    M = len(valid_rows)
    if M < seq_len:
        return 0

    return M - seq_len + 1


#----------------data_preprocess-------------
if __name__ == "__main__":
    data1 = [0.8750861671026311, 0.9080258560447241, 0.8960526315789473, 0.9035714285714286, 0.8880457089552238, 0.8716427640156453, 0.8885930408472013, 0.9002473498233217, 0.9039514348785872, 0.9188621444201313, 0.9032784810126582, 0.8969489559164734, 0.9095687331536388, 0.9102898550724637, 0.907581081081081, 0.9061604584527221, 0.9130694444444446, 0.9459532710280374, 0.9475409403063919, 0.916459807073955]

    data2 = [0.9631981843360518, 0.9608944793850456, 0.951374269005848, 0.9584982578397213, 0.9453311567164179, 0.9531616688396349, 0.9575416036308624, 0.9753091872791522, 0.9796136865342161, 0.9781509846827133, 0.978, 0.9793619489559164, 0.9841644204851752, 0.9879130434782609, 0.9904594594594596, 0.9917191977077364, 0.9938472222222222, 0.9962242990654206, 0.9965755414685684, 0.9959839228295819]

    ave1 = sum(data1) / len(data1)
    ave2 = sum(data2) / len(data2)
    print((ave1 + ave2) / 2) 
