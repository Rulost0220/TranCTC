import os
import numpy as np
import pandas as pd
import argparse


def add_noise_to_csvs_in_dir(input_dir: str, tau: float, x: float, seed: int = None) -> str:
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
    ----
    input_dir : str
        English documentation retained from the original workflow.
    tau : float
        English documentation retained from the original workflow.
    x : float
        English documentation retained from the original workflow.
    seed : int, optional
        English documentation retained from the original workflow.

    English documentation retained from the original workflow.
    ----
    str
        English documentation retained from the original workflow.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if tau <= 0:
        raise ValueError("tau must be greater than 0")
    if not (0 <= x <= 1):
        raise ValueError("x must satisfy 0 <= x <= 1")

    if seed is not None:
        np.random.seed(seed)

    input_dir = os.path.abspath(input_dir)

    # English note retained from the original workflow.
    subdirs = [entry.name for entry in os.scandir(input_dir) if entry.is_dir()]
    if subdirs:
        raise ValueError(f"Input directory must not contain subdirectories, but found: {subdirs}")

    parent_dir = os.path.dirname(input_dir)
    base_name = os.path.basename(os.path.normpath(input_dir))
    output_dir = os.path.join(parent_dir, f"{base_name}_{x}_{tau}")
    os.makedirs(output_dir, exist_ok=True)

    def perturb_value(old_value: float, tau_: float) -> float:
        while True:
            new_value = old_value + np.random.normal(loc=0.0, scale=tau_)
            if new_value >= 0:
                return new_value

    def unique_output_path(dst_dir: str, filename: str) -> str:
        candidate = os.path.join(dst_dir, filename)
        while os.path.exists(candidate):
            filename = "1_" + filename
            candidate = os.path.join(dst_dir, filename)
        return candidate

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".csv"):
            continue

        src_csv = os.path.join(input_dir, fname)
        df = pd.read_csv(src_csv)

        if "interval" not in df.columns:
            raise ValueError(f"{src_csv}  does not contain an 'interval' column")

        valid_idx = df["interval"].dropna().index.to_numpy()
        n = len(valid_idx)

        if n > 0 and x > 0:
            k = int(round(n * x))
            k = min(k, n)

            if k > 0:
                chosen_idx = np.random.choice(valid_idx, size=k, replace=False)
                for idx in chosen_idx:
                    old_val = float(df.at[idx, "interval"])
                    df.at[idx, "interval"] = perturb_value(old_val, tau)

        dst_csv = unique_output_path(output_dir, fname)
        df.to_csv(dst_csv, index=False)

    return output_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add noise to reference normal flows")
    parser.add_argument("--input_dir", default="./normal_dir")
    parser.add_argument("--tau", type=float, default=10)
    parser.add_argument("--x", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = add_noise_to_csvs_in_dir(args.input_dir, args.tau, args.x, args.seed)
    print("Output directory:", out_dir)
