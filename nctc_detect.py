import torch
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
# from nctc_window_detect import detect_nctc
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score
from scipy.stats import ks_2samp, entropy
from ks_detect import *
import pdb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import tempfile
import gzip
from Transformer.model import *
from GAS.model import *
from GAS.train import *
from thop import profile
from scipy.fftpack import dct
# from GAS import model, train
FALSE_POSITIVE_RANGE = (0, 100)
KS_NORMAL_LIST = []

def count_predictions(reference_scores, test_scores, low_q=0, high_q=95):
    """
    根据 reference_scores 的分位数阈值，统计 test_scores 中：
    - 总文件数
    - 被判为阳性的文件数
    - 被判为阴性的文件数
    """
    ref = np.array(reference_scores, dtype=float)
    test = np.array(test_scores, dtype=float)

    low = np.percentile(ref, low_q)
    high = np.percentile(ref, high_q)

    positive_mask = (test < low) | (test > high)
    positive_count = int(np.sum(positive_mask))
    total_count = int(len(test))
    negative_count = total_count - positive_count

    return total_count, positive_count, negative_count


def collect_flow_dirs(root_dir):
    """
    递归收集 root_dir 下所有目录名以 'flow' 结尾的子目录。
    如果 root_dir 自身就以 'flow' 结尾，也会被纳入。
    """
    flow_dirs = []

    root_dir = os.path.normpath(root_dir)
    if os.path.isdir(root_dir) and os.path.basename(root_dir).endswith("flow"):
        flow_dirs.append(root_dir)

    for current_root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d.endswith("flow"):
                flow_dirs.append(os.path.join(current_root, d))

    return sorted(set(flow_dirs))
    
def count_model_size(model):
    """
    计算模型的大小（单位：MB）
    """
    # 1. 计算参数（Parameters）的大小
    # param.nelement() 是参数总数，param.element_size() 是单个元素占用的字节数（float32为4）
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    
    # 2. 计算缓冲区（Buffers）的大小（如 LayerNorm 或 BatchNorm 的运行统计量）
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    # 换算单位
    size_all_bytes = param_size + buffer_size
    size_all_kb = size_all_bytes / 1024
    size_all_mb = size_all_kb / 1024

    print("-" * 30)
    print(f"Model Size Evaluation:")
    print(f"Total Bytes: {size_all_bytes} Bytes")
    print(f"Size in KB:  {size_all_kb:.2f} KB")
    print(f"Size in MB:  {size_all_mb:.4f} MB")
    print("-" * 30)
    
    return size_all_mb

# 使用方法：
# model = Transformer_packets_feature_model(...)
# count_model_size(model)


def get_detection_result_anomaly(reference_scores: list, test_scores: list, threshold_p: int):
    # This method outputs the TPR (true-positive rate) and FPR (false-positive rate) of detection.

    threshold_up = np.percentile(reference_scores, threshold_p)
    threshold_down = np.percentile(reference_scores, 100 - threshold_p)

    legit_test_count = np.sum(test_scores < threshold_up)
    illegit_test_count = len(test_scores) - legit_test_count
    TP_rate_up = illegit_test_count / len(test_scores)

    legit_test_count = np.sum(test_scores > threshold_down)
    illegit_test_count = len(test_scores) - legit_test_count
    TP_rate_down = illegit_test_count / len(test_scores)

    return max(TP_rate_up, TP_rate_down)

def get_AUC(valiset_losses: list, testset_losses: list):
    TPR_scores = [get_detection_result_anomaly(reference_scores = valiset_losses,
                                                  test_scores = testset_losses,
                                                  threshold_p = i) for i in range(100 - FALSE_POSITIVE_RANGE[0], 100 - FALSE_POSITIVE_RANGE[1] - 1, -1)]
    FPR_scores = [i / 100 for i in range(FALSE_POSITIVE_RANGE[0], FALSE_POSITIVE_RANGE[1] + 1)]
    AUC = auc(FPR_scores, TPR_scores)
    return AUC

def get_FPR(reference, test, low_q=5, high_q=95):
    """
    reference: list1 (合法参考)
    test: list2 (待测，其实也合法)
    low_q, high_q: 分位数阈值 (默认5%和95%)
    """
    ref = np.array(reference, dtype=float)
    test = np.array(test, dtype=float)

    # 定义正常区间
    low = np.percentile(ref, low_q)
    high = np.percentile(ref, high_q)

    # 统计 test 中被误判为异常的比例
    fp_count = np.sum((test < low) | (test > high))
    fpr = fp_count / len(test)
    return fpr


def scores_calculate(flow_num, model_type, scores_mode, model, criterion, csv_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle, masked_idx=-1):
    model.eval()
    device = next(model.parameters()).device
    csv_files = []

    if scores_mode == 'reference':
        for root, _, files in os.walk(csv_dir):
            for file in files:
                if file.lower().endswith(".csv"):
                    name, _ = os.path.splitext(file)  # "12.csv" -> "12"
                    if name.isdigit() and int(name) < 7500:   # 只要 0.csv ~ (n-1).csv
                        csv_files.append(os.path.join(root, file))

    else:
        if os.path.isfile(csv_dir) and csv_dir.endswith(".csv"):
            csv_files.append(csv_dir)
        elif os.path.isdir(csv_dir):
            num = 0
            for root, dirs, files in os.walk(csv_dir):
                if num >= flow_num:
                    break
                for file in files:
                    if num >= flow_num:
                        break
                    if file.endswith(".csv"):
                        num += 1
                        csv_files.append(os.path.join(root, file))



    scores_list = []
    for csv_file in csv_files:
        data_loader = get_dataloader(model_type, csv_file, feature_dim, seq_len, median, num_samples, batch_size, is_middle, masked_idx)
        loss_seq = 0
        total_result = 0
        total_batch_num = 0
        with torch.no_grad():
            total_num = 0
            # 计算下下述代码的执行时间
            for batch_x, batch_y in data_loader:
                # print(batch_x.shape)
                # pdb.set_trace()
                batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
#                flops, params = profile(model, inputs=(batch_x, ))
#                count_model_size(model)
#                print(f"FLOPs: {flops / 1e6:.2f} M")
#                print(f"Params: {params / 1e6:.2f} M")

#                pdb.set_trace()
                if masked_idx != -1:
                    batch_x[:, :, masked_idx] = 0
                t0 = time.time()
                output, _, _= model(batch_x)
                torch.cuda.synchronize()
                t1 = time.time()
                throughput = batch_size / (t1 - t0)
                total_result += throughput
                total_batch_num += 1
#                print(f"{throughput} windows / s")
                loss = criterion(output, batch_y)
                loss_seq += loss.item() * batch_y.size(0)
                total_num += batch_y.size(0)
#        print(f" Average {total_result/total_batch_num} windows / s")
#        pdb.set_trace()
        scores_list.append(loss_seq / total_num)
    return scores_list

#
#    scores_list = []
#    
#    starter = torch.cuda.Event(enable_timing=True)
#    ender = torch.cuda.Event(enable_timing=True)
#    batch_size = 1
#    for csv_file in csv_files:
#        data_loader = get_dataloader(
#            model_type,
#            csv_file,
#            feature_dim,
#            seq_len,
#            median,
#            num_samples,
#            batch_size,
#            is_middle,
#            masked_idx
#        )
#    
#        with torch.no_grad():
#            batch_x, batch_y = next(iter(data_loader))
#            batch_x = batch_x.to(device)
#            for _ in range(50):
#                _, _, _ = model(batch_x)
#        torch.cuda.synchronize()
#    
#        loss_seq = 0
#        total_num = 0
#        with torch.no_grad():
#            total_num = 0
#            total_time = 0
#            # 计算下下述代码的执行时间
#            for batch_x, batch_y in data_loader:
#                # print(batch_x.shape)
#                # pdb.set_trace()
#                batch_x = batch_x.to(device)
#                batch_y = batch_y.long().to(device)
#    
#                starter.record()
#                output, _, _ = model(batch_x)
#                ender.record()
#    
#                torch.cuda.synchronize()
#                elapsed_ms = starter.elapsed_time(ender)
#                print(f"{elapsed_ms} ms / window")
#                total_time += elapsed_ms
#                total_num += 1    
#                loss = criterion(output, batch_y)
#                loss_seq += loss.item() * batch_y.size(0)
#                total_num += batch_y.size(0)
#                
#            print(f"Averge:{total_time/total_num} ms / window")
#            pdb.set_trace()
#    
#        scores_list.append(loss_seq / total_num)
#    
#    return scores_list


# gas 方法（纠错后）
def gas_detect(model, criterion, normal_dir, flow_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle):
    # 已经得到了两个损失序列
    # 遍历 p 从 1 到 100，计算对应阈值下的 AUC

    # 判断normal_dir是否是一个 list
    if isinstance(normal_dir, list):
        gas_normal_list = normal_dir
    else:
        gas_normal_list = scores_calculate('GAS', 'reference', model, criterion, normal_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle=False)

    if isinstance(flow_dir, list):
        gas_nctc_list = flow_dir
    else:
        gas_nctc_list = scores_calculate('GAS', 'detect', model, criterion, flow_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle=False)
    FPR_list = []
    TPR_list = []
    FPR = get_FPR(gas_normal_list, gas_nctc_list, low_q=5, high_q=95)
    AUC = get_AUC(gas_normal_list,gas_nctc_list)
    return (AUC, FPR)

def transformer_ipd_detect(model, criterion, normal_dir, flow_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle):
    # 已经得到了两个损失序列
    # 遍历 p 从 1 到 100，计算对应阈值下的 AUC

    # 判断normal_dir是否是一个 list
    if isinstance(normal_dir, list):
        gas_normal_list = normal_dir
    else:
        gas_normal_list = scores_calculate('Transformer_ipd_model', 'reference', model, criterion, normal_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle)

    if isinstance(flow_dir, list):
        gas_nctc_list = flow_dir
    else:
        gas_nctc_list = scores_calculate('Transformer_ipd_model', 'detect', model, criterion, flow_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle)
    FPR_list = []
    TPR_list = []

    AUC = get_AUC(gas_normal_list,gas_nctc_list)
    return AUC

def transformer_packet_detect(flow_num, model, criterion, normal_dir, flow_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle, masked_idx=-1):
    # 已经得到了两个损失序列
    # 遍历 p 从 1 到 100，计算对应阈值下的 AUC

    # 判断normal_dir是否是一个 list
    if isinstance(normal_dir, list):
        gas_normal_list = normal_dir
    else:
        gas_normal_list = scores_calculate(flow_num, 'Transformer_packets_feature_model', 'reference', model, criterion, normal_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle, masked_idx)

    if isinstance(flow_dir, list):
        gas_nctc_list = flow_dir
    else:
        gas_nctc_list = scores_calculate(flow_num, 'Transformer_packets_feature_model', 'detect', model, criterion, flow_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle, masked_idx)
    FPR_list = []
    TPR_list = []

    AUC = get_AUC(gas_normal_list,gas_nctc_list)
    return AUC

#类似于用未知流量和正常流量做聚类，差距大的就是异常流量
def ks_detect_fast(normal_dir, flow_dir, seq_len, kfold=5, use_hist=False, n_bins=256, baseline_cap=None):
    # ---- 读入数据 ----
    normal_ipd_list, nctc_ipd_list = [], []

    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)
                if name.isdigit() and int(name) < 15000:
                    normal_ipd_list.append(ipd_csv_to_ipd_list(os.path.join(root, file), seq_len))
    print('正常序列数量:', len(normal_ipd_list))

    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                nctc_ipd_list.append(ipd_csv_to_ipd_list(os.path.join(root, file), seq_len))
    print('未知序列数量:', len(nctc_ipd_list))

    # ---- （可选）限制基线样本数，防止内存/排序过慢 ----
    if baseline_cap is not None:
        # 随机抽样合并，cap 表示总样本上限（而非窗口数）
        all_norm = np.concatenate(normal_ipd_list)
        if len(all_norm) > baseline_cap:
            sel = np.random.choice(len(all_norm), size=baseline_cap, replace=False)
            all_norm = all_norm[sel]
            # 用回切分后的等长块替代原序列（只为兼容，正常/未知窗口仍各自评估）
            chunk = len(normal_ipd_list[0])
            normal_ipd_list = [all_norm[i:i+chunk] for i in range(0, len(all_norm)-chunk+1, chunk)]
            print(f'基线样本下采样到 {len(all_norm)}')

    # ---- 1) 计算 ks_normal_list：K折“留一块做基线”，避免 O(N^2) ----
    N = len(normal_ipd_list)
    folds = kfold_indices(N, k=max(2, kfold))
    ks_normal_list = np.empty(N, dtype=float)

    if use_hist:
        # 为每折构建直方图基线
        for fold in tqdm(folds, desc='KS (normal vs K-fold baseline, hist)'):
            comp_idx = np.setdiff1d(np.arange(N), fold, assume_unique=True)
            bins, base_cum = build_hist_baseline([normal_ipd_list[i] for i in comp_idx], n_bins=n_bins)
            for i in fold:
                ks_normal_list[i] = ks_hist_approx(normal_ipd_list[i], bins, base_cum)
    else:
        # 精确版：每折拼接并排序一次基线，然后对折内样本逐个用 searchsorted 计算
        for fold in tqdm(folds, desc='KS (normal vs K-fold baseline, exact)'):
            comp_idx = np.setdiff1d(np.arange(N), fold, assume_unique=True)
            base_sorted = np.sort(np.concatenate([normal_ipd_list[i] for i in comp_idx]))
            for i in fold:
                ks_normal_list[i] = ks_against_sorted_baseline(normal_ipd_list[i], base_sorted)

    # ---- 2) 计算 ks_nctc_list：统一与“全体正常基线”比较（一次排序，多次复用） ----
    if use_hist:
        bins, base_cum = build_hist_baseline(normal_ipd_list, n_bins=n_bins)
        ks_nctc_list = [ks_hist_approx(x, bins, base_cum) for x in tqdm(nctc_ipd_list, desc='KS (nctc vs baseline, hist)')]
    else:
        base_sorted_all = np.sort(np.concatenate(normal_ipd_list))
        ks_nctc_list = [ks_against_sorted_baseline(x, base_sorted_all) for x in tqdm(nctc_ipd_list, desc='KS (nctc vs baseline, exact)')]

    # ---- 指标（沿用你的评估函数）----
    FPR = get_FPR(ks_normal_list, ks_nctc_list, low_q=5, high_q=95)
    AUC = get_AUC(ks_normal_list, ks_nctc_list)
    return (AUC, FPR)

def ks_detect(normal_dir, flow_dir, seq_len):
    normal_ipd_list = []
    nctc_ipd_list = []

    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)  # "12.csv" -> "12"
                if name.isdigit() and int(name) < 15000:   # 只要 0.csv ~ (n-1).csv
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
    print('正常序列数量:', len(normal_ipd_list))

    # 遍历 flow_dir 中的所有 CSV 文件，加载 IPD 数据
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    print('未知序列数量:', len(nctc_ipd_list))

    # 计算 ks_normal_list（每个正常序列与其余正常序列的平均 K-S 值）
    if len(KS_NORMAL_LIST) == 0: 
        for i in tqdm(range(len(normal_ipd_list)), desc="Calculating KS for normal sequences"):
            ks_vals = []
            for j in range(len(normal_ipd_list)):
                if i != j:
                    ks_stat, _ = ks_2samp(normal_ipd_list[i], normal_ipd_list[j])
                    ks_vals.append(ks_stat)
            KS_NORMAL_LIST.append(np.mean(ks_vals))
        
    # 计算 ks_nctc_list（每个未知序列与每一个正常序列的平均 K-S 值）
    ks_nctc_list = []
    for nctc_seq in tqdm(nctc_ipd_list, desc="Calculating KS for nctc sequences"):
        ks_vals = []
        for normal_seq in normal_ipd_list:
            ks_stat, _ = ks_2samp(nctc_seq, normal_seq)
            ks_vals.append(ks_stat)
        ks_nctc_list.append(np.mean(ks_vals))

    # 遍历 p 从 1 到 100，计算对应阈值下的 AUC
    FPR_list = []
    TPR_list = []
    
    FPR = get_FPR(ks_normal_list, ks_nctc_list, low_q=5, high_q=95)

    AUC = get_AUC(ks_normal_list,ks_nctc_list)
    return (AUC, FPR)

# deviation_detect 计算窗口间标准差的变化模式，越小，说明流量越规律，更有可能是异常流量
def deviation_detect(normal_dir, flow_dir, seq_len):
    normal_ipd_list = []
    nctc_ipd_list = []


    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)  # "12.csv" -> "12"
                if name.isdigit() and int(name) < 15000:   # 只要 0.csv ~ (n-1).csv
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
    # print('正常序列数量:', len(normal_ipd_list))
    # 遍历 flow_dir 中的所有 CSV 文件，加载 IPD 数据
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
    # print('未知序列数量:', len(nctc_ipd_list))
    # 计算 normal_list 中每个正常序列的 Reg值
    normal_reg_list = []
    for normal_seq in normal_ipd_list:
        values = []
        # 每 50个数据划分为一个 segment，记录所有的 segment
        segments = [normal_seq[i:i + 50] for i in range(0, len(normal_seq) - 49 , 10)]
        for i in range(len(segments)):
            sigma_i = np.std(segments[i])
            for j in range(i + 1, len(segments)):
                sigma_j = np.std(segments[j])
                if sigma_i != 0:  # 避免除以0
                    diff = abs((sigma_i - sigma_j) / sigma_i)
                    values.append(diff)
        normal_seq_reg = np.mean(values)
        normal_reg_list.append(normal_seq_reg)

    # 计算 nctc_list 中每个序列的 Reg值
    nctc_reg_list = []
    for nctc_seq in nctc_ipd_list:
        values = []
        segments = [nctc_seq[i:i + 50] for i in range(0, len(nctc_seq) - 49, 10)]
        for i in range(len(segments)):
            sigma_i = np.std(segments[i])
            for j in range(i + 1, len(segments)):
                sigma_j = np.std(segments[j])
                if sigma_i != 0:  # 避免除以0
                    diff = abs((sigma_i - sigma_j) / sigma_i)
                    values.append(diff)
        nctc_seq_reg = np.mean(values)
        nctc_reg_list.append(nctc_seq_reg)

    # print(f"正常序列的 Reg 值列表: {normal_reg_list}")
    # print(f"未知序列的 Reg 值列表: {nctc_reg_list}")

    FPR = get_FPR(normal_reg_list, nctc_reg_list, low_q=5, high_q=95)
    AUC = get_AUC(normal_reg_list, nctc_reg_list)
    return (AUC,FPR)

# en_detect,计算熵值，如果测试得分低，说明样本可能是异常流量
def en_detect(normal_dir, flow_dir, seq_len, bins=64):
    normal_ipd_list = []
    nctc_ipd_list = []
    # 遍历 normal_dir 中的所有 CSV 文件，加载 IPD 数据
    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)  # "12.csv" -> "12"
                if name.isdigit() and int(name) < 15000:   # 只要 0.csv ~ (n-1).csv
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    # 遍历 flow_dir 中的所有 CSV 文件，加载 IPD 数据
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    # 离散化
    entropies_normal = []
    for normal_seq in normal_ipd_list:
        hist, _ = np.histogram(normal_seq, bins=bins, density=True)
        en = entropy(hist + 1e-10, base=2)
        entropies_normal.append(en)
    
    entropies_nctc = []
    for nctc_seq in nctc_ipd_list:
        hist, _ = np.histogram(nctc_seq, bins=bins, density=True)
        en = entropy(hist + 1e-10, base=2)
        entropies_nctc.append(en)

    # print(f"正常序列的熵值列表: {entropies_normal}")
    # print(f"未知序列的熵值列表: {entropies_nctc}")
    FPR = get_FPR(entropies_normal, entropies_nctc, low_q=5, high_q=95)
    AUC = get_AUC(entropies_normal, entropies_nctc)
    return (AUC, FPR)

# cce_detect,计算熵值，如果测试得分低，说明样本可能是异常流量
def cce_detect(normal_dir, flow_dir, seq_len, m=5, bins=64):
    normal_ipd_list = []
    nctc_ipd_list = []

    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)  # "12.csv" -> "12"
                if name.isdigit() and int(name) < 15000:   # 只要 0.csv ~ (n-1).csv
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
                    
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    all_data = np.concatenate(normal_ipd_list + nctc_ipd_list)
    bin_edges = np.histogram_bin_edges(all_data, bins=bins)

    cces_normal = [compute_cce_discretized(seq, m, bin_edges) for seq in normal_ipd_list]

    cces_normal = [v for v in cces_normal if v is not None]

    cces_nctc = [compute_cce_discretized(seq, m, bin_edges) for seq in nctc_ipd_list]

    cces_nctc = [v for v in cces_nctc if v is not None]

    # print(f"正常序列的 CCE 值列表: {cces_normal}")
    # print(f"未知序列的 CCE 值列表: {cces_nctc}")
    FPR = get_FPR(cces_normal, cces_nctc, low_q=5, high_q=95)

    AUC = get_AUC(cces_normal, cces_nctc)
    return (AUC, FPR)

# epsilion_detect计算相邻 interval 的差值，分值越大，IPD 差异越小，序列更规律，有可能是异常流量
def epsilion_detect(normal_dir, flow_dir, seq_len, epsilion=0.8):
    normal_ipd_list = []
    nctc_ipd_list = []

    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)  # "12.csv" -> "12"
                if name.isdigit() and int(name) < 15000:   # 只要 0.csv ~ (n-1).csv
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
                    
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
    
    normal_epsilon_score = []
    for normal_seq in normal_ipd_list:
        normal_seq.sort()
        num = 0
        for i in range(len(normal_seq) - 1):
            r = (normal_seq[i + 1] - normal_seq[i]) / (normal_seq[i] + 1e-8)
            if r < epsilion:
                num += 1
        epsilon_score = num / (len(normal_seq) - 1)
        normal_epsilon_score.append(epsilon_score)

    nctc_epsilon_score = []
    for nctc_seq in nctc_ipd_list:
        nctc_seq.sort()
        num = 0
        for i in range(len(nctc_seq) - 1):
            r = (nctc_seq[i + 1] - nctc_seq[i]) / (nctc_seq[i] + 1e-8)
            if r < epsilion:
                num += 1
        epsilon_score = num / (len(nctc_seq) - 1)
        nctc_epsilon_score.append(epsilon_score)
    # print(f"正常序列的 epsilon 分值列表: {normal_epsilon_score}")
    # print(f"未知序列的 epsilon 分值列表: {nctc_epsilon_score}")
    FPR = get_FPR(normal_epsilon_score, nctc_epsilon_score, low_q=5, high_q=95)
    AUC = get_AUC(normal_epsilon_score, nctc_epsilon_score)
    return (AUC, FPR)

import numpy as np
import os

def enhanced_epsilon_detect(normal_dir, flow_dir, seq_len, start_eps=0.01, end_eps=1.0, steps=10):
    """
    Enhanced epsilon-similarity 实现逻辑：
    通过在 [start_eps, end_eps] 范围内多次采样 epsilon 值并取平均，
    增强对模糊化处理（如 epsilon-kalibur 攻击）的捕捉能力。
    """
    normal_ipd_list = []
    nctc_ipd_list = []

    # 1. 加载数据 (保持原逻辑)
    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)
                if name.isdigit() and int(name) < 15000:
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
                    
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    # 定义内部计算函数，用于计算单一序列的增强得分
    def calculate_enhanced_score(ipd_seq):
        if len(ipd_seq) <= 1: return 0
        ipd_seq.sort()
        
        # 计算所有相邻间隔的相对差异比率 R
        # R = (IAT_{i+1} - IAT_i) / IAT_i
        diffs = np.diff(ipd_seq)
        ratios = diffs / (np.array(ipd_seq[:-1]) + 1e-8)
        
        # 核心改进：计算多个 epsilon 阈值下的平均得分
        eps_values = np.linspace(start_eps, end_eps, steps)
        scores = []
        for eps in eps_values:
            num = np.sum(ratios < eps)
            scores.append(num / len(ratios))
        
        # 返回平均得分作为该序列的特征值
        return np.mean(scores)

    # 2. 计算正常流量的增强得分
    normal_epsilon_score = [calculate_enhanced_score(seq) for seq in normal_ipd_list]

    # 3. 计算待检测流量的增强得分
    nctc_epsilon_score = [calculate_enhanced_score(seq) for seq in nctc_ipd_list]

    # 4. 计算评估指标 (FPR, AUC)
    FPR = get_FPR(normal_epsilon_score, nctc_epsilon_score, low_q=5, high_q=95)
    AUC = get_AUC(normal_epsilon_score, nctc_epsilon_score)
    
    return (AUC, FPR)

def regularity_detect(normal_dir, flow_dir, seq_len, window_size=10):
    import numpy as np
    import os

    def compute_regularity(ipd_seq, window_size):
        # 切分为非重叠窗口
        num_windows = len(ipd_seq) // window_size
        std_list = []

        for i in range(num_windows):
            window = ipd_seq[i * window_size: (i + 1) * window_size]
            std = np.std(window)
            std_list.append(std + 1e-8)  # 避免除以 0

        # 计算 pairwise 的相对差异
        diff_list = []
        for i in range(len(std_list)):
            for j in range(i + 1, len(std_list)):
                r = abs(std_list[i] - std_list[j]) / std_list[i]
                diff_list.append(r)

        if len(diff_list) == 0:
            return 0.0  # 防止空窗口

        # 最终的 regularity 分数
        return np.std(diff_list)

    # --------------------- 加载 IPD ---------------------
    normal_ipd_list = []
    nctc_ipd_list = []

    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)
                if name.isdigit() and int(name) < 15000:
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    # ------------------- 计算 Regularity -------------------
    normal_regularity_scores = [compute_regularity(seq, window_size) for seq in normal_ipd_list]
    nctc_regularity_scores = [compute_regularity(seq, window_size) for seq in nctc_ipd_list]

    # ------------------- 返回 AUC/FPR -------------------
    FPR = get_FPR(normal_regularity_scores, nctc_regularity_scores, low_q=5, high_q=95)
    AUC = get_AUC(normal_regularity_scores, nctc_regularity_scores)
    return (AUC, FPR)


# compressibility_detect 计算压缩率，压缩率越高，说明流量越规律，有可能是异常流量
def compressibility_detect(normal_dir, flow_dir, seq_len):
    normal_ipd_list = []
    nctc_ipd_list = []

    # 加载正常样本
    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)  # "12.csv" -> "12"
                if name.isdigit() and int(name) < 15000:   # 只要 0.csv ~ (n-1).csv
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
    # 加载未知样本
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))
    
    # 计算 normal_compressed_score（正常样本的压缩率）
    normal_compressed_score = []
    for normal_seq in normal_ipd_list:
        content = ipd_to_string(normal_seq)
            # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as raw_file:
            raw_file.write(content)
            raw_file_path = raw_file.name
        # 使用 gzip 进行压缩
        compressed_path = raw_file_path + '.gz'
        with open(raw_file_path, 'rb') as f_in, gzip.open(compressed_path, 'wb', compresslevel=1) as f_out:
            f_out.writelines(f_in)
        # 获取文件大小
        raw_size = os.path.getsize(raw_file_path)
        compressed_size = os.path.getsize(compressed_path)
        # 清理临时文件
        os.remove(raw_file_path)
        os.remove(compressed_path)
        # 计算压缩率
        normal_compressed_score.append(raw_size / compressed_size)
    
    # 计算 nctc_compressed_score（未知样本的压缩率）
    nctc_compressed_score = []
    for nctc_seq in nctc_ipd_list:
        content = ipd_to_string(nctc_seq)
            # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as raw_file:
            raw_file.write(content)
            raw_file_path = raw_file.name
        # 使用 gzip 进行压缩
        compressed_path = raw_file_path + '.gz'
        with open(raw_file_path, 'rb') as f_in, gzip.open(compressed_path, 'wb', compresslevel=1) as f_out:
            f_out.writelines(f_in)
        # 获取文件大小
        raw_size = os.path.getsize(raw_file_path)
        compressed_size = os.path.getsize(compressed_path)
        # 清理临时文件
        os.remove(raw_file_path)
        os.remove(compressed_path)
        # 计算压缩率
        nctc_compressed_score.append(raw_size / compressed_size)

    # print(f"正常序列的压缩率列表: {normal_compressed_score}")
    # print(f"未知序列的压缩率列表: {nctc_compressed_score}")
    FPR = get_FPR(normal_compressed_score, nctc_compressed_score, low_q=5, high_q=95)
    AUC = get_AUC(normal_compressed_score, nctc_compressed_score)
    return (AUC, FPR)


def perceptual_hash_detect(normal_dir, flow_dir, seq_len, hash_size=8):
    """
    完全对齐 2024 Zhuang 等人论文逻辑：
    1. 归一化 (适应小数)
    2. 二维重塑 (图像化)
    3. 2D-DCT 变换
    4. 汉明距离计算相似度
    """

    def compute_phash_strictly(ipd_seq, hash_size):
        # A. 预处理：映射到 0-255 像素空间 (解决你关心的小数影响问题)
        ipd_array = np.array(ipd_seq)
        min_v, max_v = np.min(ipd_array), np.max(ipd_array)
        if max_v - min_v > 1e-9:
            pixels = ((ipd_array - min_v) / (max_v - min_v) * 255).astype(np.float32)
        else:
            pixels = np.zeros_like(ipd_array)

        # B. 图像化：重塑为正方形矩阵 (原文核心)
        side = int(np.sqrt(len(pixels)))
        if side * side != len(pixels):
            pixels = pixels[:side*side] # 截断对齐
        matrix = pixels.reshape((side, side))

        # C. 2D-DCT 变换
        dct_2d = dct(dct(matrix.T, norm='ortho').T, norm='ortho')

        # D. 提取左上角低频块并生成指纹
        low_freq_block = dct_2d[:hash_size, :hash_size]
        avg = np.mean(low_freq_block)
        return (low_freq_block > avg).astype(int).flatten()

    def get_phash_score(ipd_seq, hash_size):
        # 模仿原文的自相似性检测：将流平分为前后两段
        mid = len(ipd_seq) // 2
        seg1 = ipd_seq[:mid]
        seg2 = ipd_seq[mid : 2*mid]

        if len(seg1) < hash_size * hash_size:
            return 0.0

        h1 = compute_phash_strictly(seg1, hash_size)
        h2 = compute_phash_strictly(seg2, hash_size)

        # 计算汉明距离
        dist = np.mean(h1 != h2)
        
        # 【重要】转换分数逻辑：
        # 正常流量 dist 大 -> score 小
        # 异常流量 dist 小 -> score 大 (规律性强)
        return 1.0 - dist

    # --------------------- 1. 数据加载 (保持一致) ---------------------
    normal_ipd_list = []
    nctc_ipd_list = []

    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                name, _ = os.path.splitext(file)
                if name.isdigit() and int(name) < 15000:
                    file_path = os.path.join(root, file)
                    normal_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                nctc_ipd_list.append(ipd_csv_to_ipd_list(file_path, seq_len))

    # ------------------- 2. 分数计算 -------------------
    normal_phash_scores = [get_phash_score(seq, hash_size) for seq in normal_ipd_list]
    nctc_phash_scores = [get_phash_score(seq, hash_size) for seq in nctc_ipd_list]

    # ------------------- 3. 结果返回 -------------------
    FPR = get_FPR(normal_phash_scores, nctc_phash_scores, low_q=5, high_q=95)
    AUC = get_AUC(normal_phash_scores, nctc_phash_scores)
    
    return (AUC, FPR)

GAS_PARAMETERS = {
    'time_step': 8,
    'discretized_size': 16,
    'embedding_size': 100,
    'lstm_nodes': 64
}

TRANSFORMER_PARAMETERS = {
    'discretized_size': 16,
    'feature_dim': 16,
    'd_model': 100,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'is_middle': False
}

#def gas_transformer_detect(args):
#        # 进行 GAS/Transformer 验证、
#
#    GAS_PARAMETERS = {
#        'time_step': 8,
#        'discretized_size': 16,
#        'embedding_size': 100,
#        'lstm_nodes': 64
#    }
#    TRANSFORMER_PARAMETERS = {
#        'discretized_size': 16,
#        'feature_dim': args.feature_dim,
#        'd_model': args.d_model,
#        'nhead': args.nhead,
#        'num_layers': args.num_layers,
#        'dim_feedforward': args.dim_feedforward,
#        'dropout': args.dropout,
#        'is_middle': args.is_middle,
#        'masked_idx': args.masked_idx
#    }
#    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
#    print(f"Using Device{device}")
#
#    model_registry = {
#        "Transformer_packets_feature_model": (Transformer_packets_feature_model, transformer_packet_detect, TRANSFORMER_PARAMETERS),
#        "Transformer_ipd_model": (Transformer_ipd_model, transformer_ipd_detect, TRANSFORMER_PARAMETERS),
#        'GAS': (lstm_model, gas_detect, GAS_PARAMETERS)
#    }
#    criterion = nn.CrossEntropyLoss()
#    ModelCls, detect_fn, default_params = model_registry[args.model_type]
#    model = ModelCls(**default_params).to(device)
#    state_dict = torch.load(args.checkpoint_path, map_location=device)
#    model.load_state_dict(state_dict)
#    model.eval()
#    print(f"{ModelCls} loaded.")
#    
#    nctc_dirs = [d for d in os.listdir(args.nctc_root) if os.path.isdir(os.path.join(args.nctc_root, d))]
#
#    # 先计算reference_scores
#    print("Calculating reference scores...")
#    print(args.model_type)
#    print(args.is_middle)
#    normal_scores = scores_calculate(args.model_type, 'reference', model, criterion, args.normal_dir, feature_dim=args.feature_dim, seq_len=args.seq_len, median=args.median, num_samples=args.num_samples, batch_size=args.batch_size, is_middle=args.is_middle, masked_idx=args.masked_idx)
#    # print(f"参考分数列表长度: {len(normal_scores)}")
#    for nctc_dir_ in nctc_dirs:
#        if nctc_dir_:
#            nctc_dir = os.path.join(args.nctc_root, nctc_dir_)
#            print(f"Processing directory: {nctc_dir_}")
#            for sub in sorted(os.listdir(nctc_dir), key=lambda s: int(s.split('_', 1)[0])):
#                flow_subdir = os.path.join(nctc_dir, sub)
#                result = detect_fn(model, criterion, normal_scores, flow_subdir, feature_dim=args.feature_dim, seq_len=args.seq_len, median=args.median, num_samples=args.num_samples, batch_size=args.batch_size, is_middle=args.is_middle, masked_idx=args.masked_idx)
#                if isinstance(result, tuple):
#                    AUC, FPR = result
#                    print(f"{sub} AUC: {AUC}")
#                else:
#                    AUC = result
#                    print(f"{sub} AUC: {AUC}")
def gas_transformer_detect(args):
    GAS_PARAMETERS = {
        'time_step': 8,
        'discretized_size': 16,
        'embedding_size': 100,
        'lstm_nodes': 64
    }
    TRANSFORMER_PARAMETERS = {
        'discretized_size': 16,
        'feature_dim': args.feature_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'is_middle': args.is_middle,
        'masked_idx': args.masked_idx
    }

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device {device}")

    model_registry = {
        "Transformer_packets_feature_model": (Transformer_packets_feature_model, TRANSFORMER_PARAMETERS),
        "Transformer_ipd_model": (Transformer_ipd_model, TRANSFORMER_PARAMETERS),
        "GAS": (lstm_model, GAS_PARAMETERS)
    }

    criterion = nn.CrossEntropyLoss()
    ModelCls, default_params = model_registry[args.model_type]
    model = ModelCls(**default_params).to(device)

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"{ModelCls} loaded.")

    print("Calculating reference scores...")
    print(args.model_type)
    print(args.is_middle)

    normal_scores = scores_calculate(
        args.flow_num,
        args.model_type,
        'reference',
        model,
        criterion,
        args.normal_dir,
        feature_dim=args.feature_dim,
        seq_len=args.seq_len,
        median=args.median,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        is_middle=args.is_middle,
        masked_idx=args.masked_idx
    )

    flow_dir = args.nctc_root



    if flow_dir is not None:
        print(f"Processing directory: {flow_dir}")

        test_scores = scores_calculate(
            args.flow_num,
            args.model_type,
            'detect',
            model,
            criterion,
            flow_dir,
            feature_dim=args.feature_dim,
            seq_len=args.seq_len,
            median=args.median,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            is_middle=args.is_middle,
            masked_idx=args.masked_idx
        )

        total_count, positive_count, negative_count = count_predictions(
            normal_scores,
            test_scores,
            low_q=5,
            high_q=95
        )

        dir_name = os.path.basename(os.path.normpath(flow_dir))
        print(f"{dir_name} -> total: {total_count}, positive: {positive_count}, negative: {negative_count}")
def other_method_detect(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device{device}")

    # model_registry = {
    # 	"deviation_detect": deviation_detect,
    # 	"en_detect": en_detect,
    # 	"cce_detect": cce_detect,
    # 	"epsilion_detect": epsilion_detect,
    # 	"compressibility_detect": compressibility_detect,
    # }

    
#    model_registry = {
#        "en_detect": en_detect,
#        "cce_detect": cce_detect,
#        "deviation_detect": deviation_detect,
#        "epsilion_detect": epsilion_detect,
#        "compressibility_detect": compressibility_detect,
#        "regularity_detect": regularity_detect
#    }

    model_registry = {"perceptual_hash_detect": perceptual_hash_detect}
    seq_len = None
    for method_name, detect_fn in model_registry.items():
        print(f'-----------------------{method_name}------------------------')
        nctc_dirs = [d for d in os.listdir(args.nctc_root) if os.path.isdir(os.path.join(args.nctc_root, d))]
        # 先计算reference_scores
        for nctc_dir_ in nctc_dirs:
            if nctc_dir_:
                nctc_dir = os.path.join(args.nctc_root, nctc_dir_)
                print(f"{nctc_dir_}")
                for sub in sorted(os.listdir(nctc_dir), key=lambda s: int(s.split('_', 1)[0])):
                    flow_subdir = os.path.join(nctc_dir, sub)

                    result = detect_fn(args.normal_dir, flow_subdir, seq_len)
                    if isinstance(result, tuple):
                        AUC, FPR = result
                        # print(f"{sub} FPR: {FPR}")
                        print(f"{sub} AUC: {AUC}")
                    else:
                        AUC = result
                        # print(f"{sub} AUC: {AUC}")


def get_args():
    parser = argparse.ArgumentParser(description="模型检测参数")
    
    # 添加常用参数
    parser.add_argument('--model_type', type=str, default='base', help='选择模型架构') #model_type = 'Transformer_ipd_model'
    parser.add_argument('--feature_dim', type=int, default=480, help='特征维度')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=3, help='Transformer 层数')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='前馈网络维度')
    parser.add_argument('--seq_len', type=int, default=9, help='序列长度') # seq_len = 8
    parser.add_argument('--checkpoint_path', type=str, help='模型权重')
    parser.add_argument('--is_middle', type=str2bool, help='是否结合上下文信息')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 概率')
    # parser.add_argument('--masked_fields', type=list, default=None, help='masked 特征')
    # checkpoint_path = './Transformer/model_pth/Transformer_ipd_model_is_middleFalse_train_num0_val_num0_epochs30/epoch_4.pth'
    # is_middle = False

    # checkpoint_path = './Transformer/model_pth/Transformer_ipd_model_is_middleTrue_train_num0_val_num0_epochs30/epoch_12.pth'
    # is_middle = True
    parser.add_argument('--num_samples', type=int, default=800000, help='测试样本数')
    parser.add_argument('--normal_dir', type=str, default='./data_emerged/train_test_flow/test_csv', help='参考分数目录')
    parser.add_argument('--nctc_root', type=str, default='./data_emerged/nctc_data/', help='检测目录')
    parser.add_argument('--median', type=float, default=1.389503)
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')

    parser.add_argument('--gpu_id', type=int, default=0, help='使用的 GPU ID（例如 0 表示 cuda:0）')
    parser.add_argument('--masked_idx', type=int, default = -1, help='masked 特征索引')
    parser.add_argument('--flow_num', type=int, help='csv 文件数')
        
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = get_args()
    if args.model_type in ['GAS', 'Transformer_ipd_model', 'Transformer_packets_feature_model']:
        # gas/transformer 方法
        gas_transformer_detect(args)
    else:
        other_method_detect(args)
        print('finished')
