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
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *




def _bucket_name(row_count: int, min_len: int, max_len: int, step: int):
    if row_count < min_len:
        return None
    if row_count >= max_len:
        return f"{max_len}_plus"
    for i in range(min_len, max_len, step):
        if i <= row_count < i + step:
            return f"{i}_{i+step}"
    return None

def clean_rebucket_intervals_with_progress(
    root_dir: str,
    min_interval: float,
    max_interval: float,
    min_len: int = 100,
    max_len: int = 2000,
    step: int = 100,
    verbose: bool = False,
):
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"'{root_dir}' is not a valid directory")

    # English note retained from the original workflow.
    split_dirs = []
    for name in ("train_csv", "test_csv"):
        p = root / name
        if p.is_dir():
            split_dirs.append(p)
    if not split_dirs:
        split_dirs = [root]  # English note retained from the original workflow.

    total_moved = total_deleted = total_removed_rows = 0
    total_processed = 0

    for split_root in split_dirs:
        # English note retained from the original workflow.
        buckets = [d for d in split_root.iterdir() if d.is_dir()]
        if not buckets:
            continue

        for bucket in buckets:
            files = list(bucket.glob("*.csv"))
            if not files:
                continue

            pbar = tqdm(files, desc=f"{split_root.name}/{bucket.name}",
                        unit="file", mininterval=0.3, leave=False)

            for csv_path in pbar:
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    if verbose:
                        print(f"[skip] failed to read {csv_path}: {e}")
                    continue

                if "interval" not in df.columns:
                    if verbose:
                        print(f"[skip] {csv_path} has no 'interval' column")
                    continue

                # English note retained from the original workflow.
                s = pd.to_numeric(df["interval"], errors="coerce")
                mask_keep = s.notna() & (s > min_interval) & (s < max_interval)
                removed_rows = int((~mask_keep).sum())

                if removed_rows > 0:
                    df = df.loc[mask_keep].copy()
                    try:
                        df.to_csv(csv_path, index=False)
                    except Exception as e:
                        if verbose:
                            print(f"[error] failed to write back {csv_path}: {e}")
                        continue
                    total_removed_rows += removed_rows

                # English note retained from the original workflow.
                row_count = len(df)
                new_bucket = _bucket_name(row_count, min_len, max_len, step)

                if new_bucket is None:
                    # English note retained from the original workflow.
                    try:
                        os.remove(csv_path)
                        total_deleted += 1
                        if verbose:
                            print(f"[delete] {csv_path} rows {row_count} < {min_len}")
                    except Exception as e:
                        if verbose:
                            print(f"[error] failed to delete {csv_path}: {e}")
                    total_processed += 1
                    continue

                # English note retained from the original workflow.
                if bucket.name != new_bucket:
                    dest_dir = split_root / new_bucket
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / csv_path.name
                    try:
                        os.replace(csv_path, dest)  # English note retained from the original workflow.
                        total_moved += 1
                        if verbose:
                            print(f"[move] {csv_path} -> {dest} (rows {row_count})")
                    except Exception as e:
                        if verbose:
                            print(f"[error] failed to move {csv_path}: {e}")
                else:
                    if verbose:
                        action = "kept" if removed_rows == 0 else "cleaned in place and kept"
                        print(f"[{action}] {csv_path}  rows={row_count}  removed rows={removed_rows}")

                total_processed += 1
            pbar.close()

    print(f"Done: processed {total_processed} files, moved {total_moved}, deleted {total_deleted}, removed rows total {total_removed_rows}.")


def load_timestamps(times_path):
    with open(times_path, "r") as f:
        return [float(line.strip()) for line in f]

def process_pcap_with_times(pcap_path, times_path):
    timestamps = load_timestamps(times_path)

    # English note retained from the original workflow.
    total_packets = sum(1 for _ in RawPcapReader(pcap_path))

    # English note retained from the original workflow.
    packets = []
    with tqdm(RawPcapReader(pcap_path), total=total_packets, mininterval=60,
              desc=f"Processing {os.path.basename(pcap_path)}") as pbar:
        for pkt in pbar:
            packets.append(pkt)

    if len(packets) != len(timestamps):
        raise ValueError("pcap packet count does not match the number of rows in the times file!")

    return packets, timestamps

def extract_ipv6_flows(packets, timestamps, output_dir):
    flows = defaultdict(list)
    os.makedirs(output_dir, exist_ok=True)

    for i, (pkt_data, _) in enumerate(packets):
        ipv6_pkt = parse_ipv6_packet(pkt_data, i+1)
        if ipv6_pkt and TCP in ipv6_pkt:
            tcp = ipv6_pkt[TCP]
            flow_key = (
                ipv6_pkt.src, tcp.sport,
                ipv6_pkt.dst, tcp.dport,
                6  # protocol number for TCP
            )

            flow_data = {
                "timestamp": timestamps[i],
                "version": int(ipv6_pkt.version),
                "tc": int(ipv6_pkt.tc),
                "fl": int(ipv6_pkt.fl),
                "plen": int(ipv6_pkt.plen),
                "nh": int(ipv6_pkt.nh),
                "hlim": int(ipv6_pkt.hlim),
                "src": ipv6_pkt.src,
                "dst": ipv6_pkt.dst,
                "tcp_sport": int(tcp.sport),
                "tcp_dport": int(tcp.dport),
                "tcp_seq": int(tcp.seq),
                "tcp_ack": int(tcp.ack),
                "tcp_dataofs": int(tcp.dataofs),
                "tcp_reserved": int(tcp.reserved),
                "tcp_flags": int(tcp.flags),
                "tcp_window": int(tcp.window),
                "tcp_chksum": int(tcp.chksum),
                "tcp_urgptr": int(tcp.urgptr),
            }

            flows[flow_key].append(flow_data)

    # English note retained from the original workflow.
    flow_num = 0
    for idx, (flow_key, records) in enumerate(flows.items()):
        flow_name = f"flow_{idx}_{flow_key[0]}_{flow_key[1]}_{flow_key[2]}_{flow_key[3]}.csv"
        path = os.path.join(output_dir, flow_name.replace(":", "_"))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
            flow_num += 1
    print(f"Saved {flow_num} flows")

def timestamp_to_interval(csv_path):  # English note retained from the original workflow.
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # English note retained from the original workflow.
        a, b = tee(reader)
        next(b, None)  # English note retained from the original workflow.

        # English note retained from the original workflow.
        new_rows = []

        for a_row, b_row in zip(a, b):
            a_timestamp = float(a_row[0])
            b_timestamp = float(b_row[0])
            interval = (b_timestamp - a_timestamp) * 1000
            a_row[0] = str(interval)
            new_rows.append(a_row)

    # English note retained from the original workflow.
    if header[0].lower() == "timestamp":
        header[0] = "interval"

    # English note retained from the original workflow.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_rows)

def split_csv_by_length(input_dir, output_dir, min_len=100, max_len=2000, step=100, train_ratio=0.7):
    """
    Group CSV files by row count and split into train/test sets with subdirectories.
    
    Parameters:
    input_dir (str): Directory containing input CSV files
    output_dir (str): Directory to store output train/test files
    min_len (int): Minimum row count for grouping
    max_len (int): Maximum row count for grouping
    step (int): Step size for grouping
    train_ratio (float): Ratio of files to be used for training
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # English note retained from the original workflow.
    csv_files = list(input_path.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return
    
    # English note retained from the original workflow.
    length_groups = {f"{i}_{i+step}": [] for i in range(min_len, max_len, step)}
    length_groups[f"{max_len}_plus"] = []
    
    # English note retained from the original workflow.
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            row_count = len(df)
            
            # English note retained from the original workflow.
            if row_count >= max_len:
                length_groups[f"{max_len}_plus"].append(file)
            else:
                for i in range(min_len, max_len, step):
                    if i <= row_count < i + step:
                        length_groups[f"{i}_{i+step}"].append(file)
                        break
        except Exception as e:
            print(f"Error reading {file.name}: {str(e)}")
            continue
    
    # English note retained from the original workflow.
    # English note retained from the original workflow.
    for group, files in length_groups.items():
        print(f"{group}: {len(files)} files")
    
    # English note retained from the original workflow.
    train_dir = output_path / "train_csv"
    test_dir = output_path / "test_csv"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # English note retained from the original workflow.
    for group, files in length_groups.items():
        if not files:
            continue
            
        # English note retained from the original workflow.
        train_group_dir = train_dir / group
        test_group_dir = test_dir / group
        train_group_dir.mkdir(parents=True, exist_ok=True)
        test_group_dir.mkdir(parents=True, exist_ok=True)
        
        # English note retained from the original workflow.
        train_files, test_files = train_test_split(files, train_size=train_ratio, random_state=42)
        
        # English note retained from the original workflow.
        for file in train_files:
            shutil.copy2(file, train_group_dir / file.name)
        
        # English note retained from the original workflow.
        for file in test_files:
            shutil.copy2(file, test_group_dir / file.name)
        
        print(f"Group {group}: {len(train_files)} training files, {len(test_files)} test files")

    print(f"Processed {len(csv_files)} CSV files and copied them to:")
    print(f" - training directory: {train_dir}, contains {len(list(train_dir.glob('**/*.csv')))} files")
    print(f" - test directory: {test_dir}, contains {len(list(test_dir.glob('**/*.csv')))} files")

def filter_short_flows(csv_dir, save_dir, n, max_interval):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.

    Args:
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.

    Returns:
        None
    """
    if not os.path.isdir(csv_dir):
        raise ValueError(f"'{csv_dir}' is not a valid directory")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    deleted_count = 0
    copied_count = 0

    for dirpath, _, filenames in os.walk(csv_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                full_path = os.path.join(dirpath, filename)
                try:
                    with open(full_path, "r") as f:
                        reader = list(csv.reader(f))
                        if not reader:
                            continue
                        header = reader[0]
                        if "interval" not in header:
                            print(f"Skipping {full_path}: no 'interval' column found.")
                            continue
                        interval_idx = header.index("interval")
                        
                        # English note retained from the original workflow.
                        filtered_rows = [
                            row for row in reader[1:]
                            if len(row) > interval_idx and float(row[interval_idx]) <= max_interval
                        ]

                        if len(filtered_rows) < n:
                            if save_dir is None:
                                os.remove(full_path)
                                deleted_count += 1
                                print(f"Deleted {full_path} (< {n} rows or interval > 1000)")
                            continue

                    if save_dir is not None:
                        relative_path = os.path.relpath(full_path, csv_dir)
                        output_path = os.path.join(save_dir, relative_path)
                        output_dir = os.path.dirname(output_path)
                        Path(output_dir).mkdir(parents=True, exist_ok=True)

                        with open(output_path, "w", newline="") as out_f:
                            writer = csv.writer(out_f)
                            writer.writerow(header)
                            writer.writerows(filtered_rows)
                        copied_count += 1
                        print(f"Copied {full_path} to {output_path} (>= {n} rows, cleaned)")
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
                    continue

    if save_dir is None:
        print(f"✅ Deleted {deleted_count} CSV files with < {n} rows or invalid interval > 1000")
    else:
        print(f"✅ Copied {copied_count} cleaned CSV files with ≥ {n} rows to {save_dir}")

def csv_to_seq_csv(csv_dir, seq_len, seq_num, output_dir, isTest=False):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    
    Args:
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
        English documentation retained from the original workflow.
    
    Returns:
        None
    """
    seq_len = seq_len + 2  # English note retained from the original workflow.
    csv_dir_path = Path(csv_dir)
    output_path = Path(output_dir)
    
    if not csv_dir_path.is_dir():
        raise ValueError(f"csv_dir '{csv_dir}' is not a valid directory")
    os.makedirs(output_dir, exist_ok=True)

    # English note retained from the original workflow.
    csv_files = list(csv_dir_path.rglob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    if not isTest:
        if seq_len < 1:
            raise ValueError("seq_len must be a positive integer")
        if seq_num < 0:
            raise ValueError("seq_num must be a non-negative integer")
        
        # English note retained from the original workflow.
        output_subdir = f"seq_len_{seq_len}_seq_num_{'all' if seq_num == 0 else seq_num}"
        output_subpath = output_path / output_subdir
        os.makedirs(output_subpath, exist_ok=True)

        # English note retained from the original workflow.
        used_sequences = set()
        generated_count = 0
        max_attempts = seq_num * 10 if seq_num > 0 else float('inf')  # English note retained from the original workflow.

        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            try:
                # English note retained from the original workflow.
                df = pd.read_csv(csv_file)
                if len(df) < seq_len:  # English note retained from the original workflow.
                    continue
                
                # English note retained from the original workflow.
                data = df.values
                max_start = len(data) - seq_len + 1
                
                if seq_num == 0:
                    # English note retained from the original workflow.
                    for start_idx in range(max_start):
                        sequence = data[start_idx:start_idx + seq_len]
                        
                        # English note retained from the original workflow.
                        seq_hash = hashlib.md5(sequence.tobytes()).hexdigest()
                        if seq_hash in used_sequences:
                            continue
                        
                        # English note retained from the original workflow.
                        used_sequences.add(seq_hash)
                        
                        # English note retained from the original workflow.
                        output_file = output_subpath / f"{generated_count + 1}.csv"
                        seq_df = pd.DataFrame(sequence, columns=df.columns)
                        seq_df.to_csv(output_file, index=False)
                        
                        generated_count += 1
                else:
                    # English note retained from the original workflow.
                    attempts = 0
                    while generated_count < seq_num and attempts < max_attempts:
                        if max_start <= 0:
                            break
                        start_idx = random.randint(0, max_start - 1)
                        sequence = data[start_idx:start_idx + seq_len]
                        
                        # English note retained from the original workflow.
                        seq_hash = hashlib.md5(sequence.tobytes()).hexdigest()
                        if seq_hash in used_sequences:
                            attempts += 1
                            continue
                        
                        # English note retained from the original workflow.
                        used_sequences.add(seq_hash)
                        
                        # English note retained from the original workflow.
                        output_file = output_subpath / f"{generated_count + 1}.csv"
                        seq_df = pd.DataFrame(sequence, columns=df.columns)
                        seq_df.to_csv(output_file, index=False)
                        
                        generated_count += 1
                        attempts = 0  # English note retained from the original workflow.
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        if seq_num > 0 and generated_count < seq_num:
            print(f"Warning: Only generated {generated_count} sequences out of {seq_num} due to insufficient unique data")
        print(f"✅ All sequences saved to: {output_subpath}, total: {generated_count} csv files.")
        # print(f"Generated {generated_count} sequences in {output_subpath}")
    
    else:
        file_counter = 0  # English note retained from the original workflow.
        for csv_file in csv_files:
            # English note retained from the original workflow.
            relative_path = csv_file.relative_to(csv_dir_path)
            # English note retained from the original workflow.
            subdir = relative_path.parent
            # English note retained from the original workflow.
            file_name = csv_file.stem
            
            # English note retained from the original workflow.
            output_subdir = output_path / subdir / file_name
            os.makedirs(output_subdir, exist_ok=True)
            
            try:
                # English note retained from the original workflow.
                df = pd.read_csv(csv_file)
                if len(df) < seq_len:
                    print(f"Skipping {csv_file}: insufficient data length")
                    continue

                header = df.columns.tolist()
                data = df.values
                max_start = len(data) - seq_len + 1

                for i in (range(max_start)):
                    sequence = data[i:i + seq_len]
                    # English note retained from the original workflow.
                    output_file = output_subdir / f"{i + 1}.csv"
                    seq_df = pd.DataFrame(sequence, columns=header)
                    seq_df.to_csv(output_file, index=False)
                    file_counter += 1

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        print(f"✅ All sequences saved to: {output_dir}, total: {file_counter} csv files.")

def process_csv_to_npy_11_with_interval(csv_dirs, save_dir, median):
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
    all_csv_files = []
    # English note retained from the original workflow.
    csv_file_to_root = {}
    
    for csv_dir in csv_dirs:
        csv_path = Path(csv_dir)
        if not csv_path.is_dir():
            raise ValueError(f"'{csv_dir}' is not a valid directory")
        # English note retained from the original workflow.
        for f in csv_path.rglob("*.csv"):
            if f.is_file():
                all_csv_files.append(f)
                csv_file_to_root[f] = csv_path

    if not all_csv_files:
        raise ValueError("No CSV files found in the provided directories")

    if median == 0:
        interval_diffs = []
        for csv_file in tqdm(all_csv_files, desc="Calculating median interval diff", mininterval=600):
            try:
                df = pd.read_csv(csv_file)
                if 'interval' not in df.columns or len(df) < 2:
                    continue
                intervals = df['interval'].values
                diffs = np.abs(intervals[1:] - intervals[:-1])
                interval_diffs.extend(diffs.tolist())
            except:
                continue
        if not interval_diffs:
            raise ValueError("No valid interval differences found")
        median_diff = np.median(interval_diffs)
        print(f"Median of interval differences: {median_diff:.6f}")
    else:
        median_diff = median

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    float_fields = ['tc', 'plen', 'nh', 'tcp_dataofs', 'tcp_flags', 'tcp_window']

    for csv_file in tqdm(all_csv_files, desc="Processing CSV", mininterval=60):
        try:
            df = pd.read_csv(csv_file)
            if not all(f in df.columns for f in float_fields + ['interval']):
                print(f"Skipping {csv_file.name} due to missing fields")
                continue
            if len(df) < 3:
                continue

            intervals = df['interval'].values
            float_data = []

            for i in range(1, len(df) - 1):
                row = df.iloc[i]
                try:
                    sample = [float(row['interval'])]
                    sample.extend([float(row[f]) for f in float_fields])
                except:
                    continue

                fv = float(abs(intervals[i] - intervals[i - 1]) > median_diff)
                bv = float(abs(intervals[i] - intervals[i + 1]) > median_diff)
                fs = float(intervals[i] > intervals[i - 1])
                bs = float(intervals[i + 1] > intervals[i])
                sample.extend([fv, bv, fs, bs])

                float_data.append(sample)

            if float_data:
                arr = np.array(float_data, dtype=np.float32)
                if arr.shape[1] != 11:
                    print(f"⚠️ Invalid shape {arr.shape} in {csv_file.name}, expected (:, 11)")
                    continue
                # English note retained from the original workflow.
                root_dir = csv_file_to_root[csv_file]
                relative_path = csv_file.relative_to(root_dir)
                # English note retained from the original workflow.
                output_subdir = output_dir / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                # English note retained from the original workflow.
                save_path = output_subdir / f"{csv_file.stem}.npy"
                np.save(save_path, arr)
                # print(f"✅ Saved {save_path}")
            else:
                print(f"⚠️ No valid rows in {csv_file.name}")

        except Exception as e:
            print(f"❌ Error in {csv_file.name}: {e}")

    return median_diff

def filter_http(csv_dir):
    num = 0
    for dirpath, dirnames, filenames in os.walk(csv_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path)
                if ((~df['tcp_sport'].isin([80, 443])) & (~df['tcp_dport'].isin([80, 443]))).any():
                    os.remove(file_path)
                    num = num + 1
                    print(f"File {file_path} is not HTTP traffic; deleting it")
                else:
                    continue

    print(f"Deleted {num} CSV files")

def truncate_csv_in_test_dir(test_dir):
    """
    Truncate CSV files in test subdirectories to the lower bound of their group length.
    
    Parameters:
    test_dir (str): Directory containing test CSV files in subdirectories (e.g., test_csv/100_200/)
    """
    test_path = Path(test_dir)
    
    if not test_path.exists():
        print(f"Test directory {test_dir} does not exist.")
        return
    
    # English note retained from the original workflow.
    for subdir in test_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # English note retained from the original workflow.
        try:
            lower_bound = int(subdir.name.split('_')[0])
        except (ValueError, IndexError):
            print(f"Could not parse subdirectory name {subdir.name}, skipping.")
            continue
            
        # English note retained from the original workflow.
        for file in subdir.glob("*.csv"):
            try:
                # English note retained from the original workflow.
                df = pd.read_csv(file)
                original_rows = len(df)
                
                if original_rows <= lower_bound:
                    # English note retained from the original workflow.
                    continue
                else:
                    print(f"File {file.name} rows ({original_rows}) is greater than {lower_bound}; truncation is required.")
                # English note retained from the original workflow.
                df_truncated = df.iloc[:lower_bound]
                
                # English note retained from the original workflow.
                df_truncated.to_csv(file, index=False)
                # English note retained from the original workflow.
                
            except Exception as e:
                print(f"Error while processing file {file.name}: {str(e)}")
                continue

    print("Truncation completed.")

def count_csv_by_rows(csv_dir, min_threshold, max_threshold):
    count_ge = 0  # English note retained from the original workflow.

    for root, _, files in os.walk(csv_dir):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                try:
                    df = pd.read_csv(path, header=None)
                    if min_threshold <= len(df) < max_threshold:
                        count_ge += 1
                except Exception:
                    continue  # English note retained from the original workflow.

    print(f"There are {count_ge} CSV files with row counts between {min_threshold} and {max_threshold}")

def clean_interval_rows(root_dir: str,
                        min_interval: float,
                        max_interval: float,
                        backup: bool = False,
                        verbose: bool = True) -> None:
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
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"'{root_dir}' is not a valid directory")

    total_files = 0
    cleaned_files = 0
    removed_rows_total = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue

            total_files += 1
            path = os.path.join(dirpath, filename)

            try:
                df = pd.read_csv(path)
            except Exception as e:
                if verbose:
                    print(f"[skip] failed to read {path}: {e}")
                continue

            if "interval" not in df.columns:
                if verbose:
                    print(f"[skip] {path} has no 'interval' column")
                continue

            # English note retained from the original workflow.
            s = pd.to_numeric(df["interval"], errors="coerce")

            # English note retained from the original workflow.
            mask_keep = s.notna() & (s > min_interval) & (s <= max_interval)
            removed_rows = int((~mask_keep).sum())

            if removed_rows == 0:
                if verbose:
                    print(f"[keep] {path} does not need row removal")
                continue

            df_clean = df.loc[mask_keep].copy()

            # English note retained from the original workflow.
            if backup:
                bak_path = path + ".bak"
                try:
                    shutil.copy2(path, bak_path)
                    if verbose:
                        print(f"[backup] {path} -> {bak_path}")
                except Exception as e:
                    if verbose:
                        print(f"[warning] backup failed {path}: {e}")

            # English note retained from the original workflow.
            try:
                df_clean.to_csv(path, index=False)
                cleaned_files += 1
                removed_rows_total += removed_rows
                if verbose:
                    print(f"[clean] {path} removed {removed_rows} rows, remaining {len(df_clean)} rows")
            except Exception as e:
                if verbose:
                    print(f"[error] failed to write back {path}: {e}")

    if verbose:
        print(f"Done: scanned {total_files} CSV files, cleaned {cleaned_files} files, removed {removed_rows_total} rows.")

# English note retained from the original workflow.

# input_dir = './pcap_time_dir/'
# English note retained from the original workflow.
# English note retained from the original workflow.
# pcap_files = {}
# times_files = {}

# for fname in os.listdir(input_dir):
#     if fname.endswith('.pcap'):
# English note retained from the original workflow.
#         pcap_files[base] = os.path.join(input_dir, fname)
#     elif fname.endswith('.times'):
# English note retained from the original workflow.
#         times_files[base] = os.path.join(input_dir, fname)

# English note retained from the original workflow.
# common_bases = set(pcap_files.keys()) & set(times_files.keys())

# for base in sorted(common_bases):
#     pcap_path = pcap_files[base]
#     times_path = times_files[base]
#     output_dir = f'./flow_dir/{base}/'

# English note retained from the original workflow.
#     if os.path.exists(output_dir):
# English note retained from the original workflow.
#         continue

# English note retained from the original workflow.
# English note retained from the original workflow.
#     packets, timestamps = process_pcap_with_times(pcap_path, times_path)
#     extract_ipv6_flows(packets, timestamps, output_dir)

# English note retained from the original workflow.
# missing_pcap = sorted(set(times_files) - set(pcap_files))
# missing_times = sorted(set(pcap_files) - set(times_files))

# for m in missing_pcap:
# English note retained from the original workflow.
# for m in missing_times:
# English note retained from the original workflow.

# English note retained from the original workflow.







# English note retained from the original workflow.
# English note retained from the original workflow.
# for dirpath, dirnames, filenames in os.walk('./flow_dir/'):
#     num_csv = 0
#     for filename in filenames:
#         if filename.endswith(".csv"):
#             full_path = os.path.join(dirpath, filename)
#             timestamp_to_interval(full_path)
# English note retained from the original workflow.
#             num_csv += 1
# English note retained from the original workflow.



# English note retained from the original workflow.
# English note retained from the original workflow.
# filter_short_flows('./flow_dir/',save_dir=None, n=100, max_interval=1000)


# English note retained from the original workflow.
# English note retained from the original workflow.
# filter_http('./flow_dir/')


# English note retained from the original workflow.
# min_len = 100
# max_len = 2000
# step = 100


# English note retained from the original workflow.
# count_csv_by_rows('./flow_dir/', min_threshold=min_len, max_threshold=max_len)
# English note retained from the original workflow.
# split_csv_by_length('./flow_dir/', './split_dir/', min_len=min_len, max_len=max_len, step=step, train_ratio=0.5)
# clean_rebucket_intervals_with_progress(
#     root_dir="./split_dir/",
#     min_interval=0.0,
#     max_interval=1000.0,
#     min_len=100,
#     max_len=2000,
#     step=100,
# English note retained from the original workflow.
# )

# English note retained from the original workflow.
# truncate_csv_in_test_dir('./test_dir/')

# English note retained from the original workflow.
# English note retained from the original workflow.
# csv_to_seq_csv(csv_dir='./train_dir/', seq_len=19, seq_num=0, output_dir='./train_seq_dir/', isTest=False)
# csv_to_seq_csv(csv_dir='./test_dir/', seq_len=19, seq_num=0, output_dir='./test_seq_dir/', isTest=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow preprocessing and feature generation")
    parser.add_argument("--train_seq_dir", nargs="+", default=["./train_seq_dir"])
    parser.add_argument("--test_seq_dir", nargs="+", default=["./test_seq_dir"])
    parser.add_argument("--train_npy_dir", default="./train_dir")
    parser.add_argument("--test_npy_dir", default="./val_dir")
    parser.add_argument("--median", type=float, default=0.0)
    args = parser.parse_args()

    print("-------------------------------------Start converting CSV files to NPY format...-------------------------------------")
    median_diff = process_csv_to_npy_11_with_interval(args.train_seq_dir, args.train_npy_dir, args.median)
    print("Training median_diff=", median_diff)
    process_csv_to_npy_11_with_interval(args.test_seq_dir, args.test_npy_dir, median_diff)
