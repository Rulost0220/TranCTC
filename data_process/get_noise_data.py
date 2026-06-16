import os
import numpy as np
import pandas as pd
import argparse


def add_noise_to_flow_subdirs(input_dir: str, output_dir: str, tau: float, x: float, seed: int = None):

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if tau <= 0:
        raise ValueError("tau must be greater than 0")
    if not (0 <= x <= 1):
        raise ValueError("x must satisfy 0 <= x <= 1")

    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    created_dirs = []

    def perturb_value(old_value: float, tau_: float) -> float:
        while True:
            new_value = old_value + np.random.normal(loc=0.0, scale=tau_)
            if new_value >= 0:
                return new_value

    # English note retained from the original workflow.
    for entry in os.scandir(input_dir):
        if not entry.is_dir():
            continue

        subdir_name = entry.name
        subdir_path = entry.path

        new_subdir_name = f"{subdir_name}_{x}_{tau}"
        new_subdir_path = os.path.join(output_dir, new_subdir_name)
        os.makedirs(new_subdir_path, exist_ok=True)
        created_dirs.append(new_subdir_path)

        # English note retained from the original workflow.
        for fname in os.listdir(subdir_path):
            if not fname.lower().endswith(".csv"):
                continue

            src_csv = os.path.join(subdir_path, fname)
            dst_csv = os.path.join(new_subdir_path, fname)

            df = pd.read_csv(src_csv)

            if "interval" not in df.columns:
                raise ValueError(f"{src_csv}  does not contain an 'interval' column")

            valid_idx = df["interval"].dropna().index.to_numpy()
            n = len(valid_idx)

            if n == 0 or x == 0:
                df.to_csv(dst_csv, index=False)
                continue

            k = int(round(n * x))
            k = min(k, n)

            if k > 0:
                chosen_idx = np.random.choice(valid_idx, size=k, replace=False)
                for idx in chosen_idx:
                    old_val = float(df.at[idx, "interval"])
                    df.at[idx, "interval"] = perturb_value(old_val, tau)

            df.to_csv(dst_csv, index=False)

    return created_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add noise to flows under detection")
    parser.add_argument("--input_dir", default="./nctc_dir")
    parser.add_argument("--output_dir", default="./noise_dir")
    parser.add_argument("--tau", type=float, default=10)
    parser.add_argument("--x", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    created = add_noise_to_flow_subdirs(args.input_dir, args.output_dir, args.tau, args.x, args.seed)
    print(created)
