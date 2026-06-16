import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Transformer.model import Transformer_ipd_model, Transformer_packets_feature_model
from utils import get_dataloader, str2bool


def collect_csv_files(path_or_dir: str) -> List[str]:
    if os.path.isfile(path_or_dir):
        return [path_or_dir] if path_or_dir.lower().endswith(".csv") else []
    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"Directory does not exist: {path_or_dir}")

    csv_files = []
    for root, _, files in os.walk(path_or_dir):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, filename))
    return sorted(csv_files)


def select_csv_files(path_or_dir: str, limit: int = 0, numeric_name_only: bool = False) -> List[str]:
    csv_files = collect_csv_files(path_or_dir)
    if numeric_name_only:
        csv_files = [
            path for path in csv_files
            if os.path.splitext(os.path.basename(path))[0].isdigit()
        ]
    if limit and limit > 0:
        csv_files = csv_files[:limit]
    return csv_files


def score_csv_files(
    csv_files: List[str],
    model_type: str,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    feature_dim: int,
    seq_len: int,
    median: float,
    num_samples: int,
    batch_size: int,
    is_middle: bool,
    masked_idx: int,
) -> List[float]:
    model.eval()
    device = next(model.parameters()).device
    scores = []

    for csv_file in csv_files:
        data_loader = get_dataloader(
            model_type,
            csv_file,
            feature_dim,
            seq_len,
            median,
            num_samples,
            batch_size,
            is_middle,
            masked_idx,
        )
        loss_sum = 0.0
        sample_count = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.long().to(device)
                if masked_idx != -1:
                    batch_x[:, :, masked_idx] = 0

                output, _, _ = model(batch_x)
                loss = criterion(output, batch_y)
                loss_sum += loss.item() * batch_y.size(0)
                sample_count += batch_y.size(0)

        if sample_count > 0:
            scores.append(loss_sum / sample_count)

    return scores


def count_predictions(reference_scores: List[float], test_scores: List[float], low_q: int, high_q: int):
    ref = np.array(reference_scores, dtype=float)
    test = np.array(test_scores, dtype=float)
    low = np.percentile(ref, low_q)
    high = np.percentile(ref, high_q)
    positive_mask = (test < low) | (test > high)
    positive_count = int(np.sum(positive_mask))
    total_count = int(len(test))
    return total_count, positive_count, total_count - positive_count


def collect_detect_dirs(nctc_root: str) -> List[str]:
    if os.path.isfile(nctc_root):
        return [nctc_root]
    if not os.path.isdir(nctc_root):
        raise FileNotFoundError(f"Directory does not exist: {nctc_root}")

    child_dirs = [
        os.path.join(nctc_root, name)
        for name in sorted(os.listdir(nctc_root))
        if os.path.isdir(os.path.join(nctc_root, name))
    ]
    return child_dirs or [nctc_root]


def build_model(args):
    model_registry = {
        "Transformer_packets_feature_model": Transformer_packets_feature_model,
        "Transformer_ipd_model": Transformer_ipd_model,
    }
    if args.model_type not in model_registry:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model_cls = model_registry[args.model_type]
    model = model_cls(
        discretized_size=16,
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        is_middle=args.is_middle,
    )
    if hasattr(model, "masked_idx"):
        model.masked_idx = args.masked_idx
    return model


def run_detect(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device {device}")

    model = build_model(args).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"{args.model_type} loaded.")

    criterion = nn.CrossEntropyLoss()
    reference_files = select_csv_files(
        args.normal_dir,
        limit=args.reference_limit,
        numeric_name_only=args.numeric_reference,
    )
    print(f"Reference CSV count: {len(reference_files)}")
    reference_scores = score_csv_files(
        reference_files,
        args.model_type,
        model,
        criterion,
        args.feature_dim,
        args.seq_len,
        args.median,
        args.num_samples,
        args.batch_size,
        args.is_middle,
        args.masked_idx,
    )

    for detect_dir in collect_detect_dirs(args.nctc_root):
        detect_files = select_csv_files(detect_dir, limit=args.flow_num)
        print(f"Processing: {detect_dir}")
        print(f"Detect CSV count: {len(detect_files)}")
        test_scores = score_csv_files(
            detect_files,
            args.model_type,
            model,
            criterion,
            args.feature_dim,
            args.seq_len,
            args.median,
            args.num_samples,
            args.batch_size,
            args.is_middle,
            args.masked_idx,
        )
        total_count, positive_count, negative_count = count_predictions(
            reference_scores,
            test_scores,
            low_q=args.low_q,
            high_q=args.high_q,
        )
        name = os.path.basename(os.path.normpath(detect_dir))
        print(f"{name} -> total: {total_count}, positive: {positive_count}, negative: {negative_count}")


def get_args():
    parser = argparse.ArgumentParser(description="Transformer detection parameters")
    parser.add_argument("--model_type", type=str, default="Transformer_packets_feature_model")
    parser.add_argument("--feature_dim", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=100)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--is_middle", type=str2bool, default=True, help="Whether to use context information")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples per CSV")
    parser.add_argument("--normal_dir", type=str, default="./normal_dir", help="Reference normal flow directory")
    parser.add_argument("--nctc_root", type=str, default="./nctc_dir", help="Flow directory to detect")
    parser.add_argument("--median", type=float, default=1.389503)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--masked_idx", type=int, default=-1)
    parser.add_argument("--flow_num", type=int, default=0, help="Maximum number of CSV files to detect")
    parser.add_argument("--reference_limit", type=int, default=0, help="Maximum number of reference CSV files")
    parser.add_argument("--numeric_reference", type=str2bool, default=False, help="Whether to use only reference CSV files with numeric names")
    parser.add_argument("--low_q", type=int, default=0)
    parser.add_argument("--high_q", type=int, default=99)
    return parser.parse_args()


if __name__ == "__main__":
    run_detect(get_args())
