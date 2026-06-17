# Transformer-based NCTC Detection

This repository provides a Transformer-based workflow for Network Covert Timing Channel (NCTC) detection, including traffic preprocessing, NCTC interval generation, detection-flow construction, model training, and detection evaluation.


### Environment

The dependencies can be installed with `environment.yml`:

```bash
conda env create -f environment.yml
```

The main experimental environment includes:

- Python 3.8
- PyTorch 1.11
- CUDA 11.3
- pandas, numpy, scipy, scikit-learn, scapy, tqdm

### Project Structure

```text
.
├── data_process/        # Data preprocessing, IPD generation, and flow construction scripts
├── detect/              # Transformer detection entry point
├── model/               # Transformer model, training entry point, and training script
├── nctc_intervals/      # Partial NCTC IPD data examples
├── utils.py             # Shared data loading, feature construction, and utility functions
├── environment.yml      # Conda environment configuration
└── README.md
```

### 1. Data Preprocessing

Raw CAIDA traffic should first be converted into flow-level CSV files and then transformed into sequence features for model training. The related scripts are located in `data_process/`.

Main files:

- `data_process/data_preprocess.py`: converts sequence CSV files into NPY feature data for training and validation.
- `data_process/ipd_filter.py`: computes the median of IPD differences.
- `data_process/clean_interval.py`: removes rows whose `interval` values are zero or abnormally large.
- `utils.py`: provides shared data loading, feature construction, and DataLoader utilities.

Example command:

```bash
python -u data_process/data_preprocess.py \
  --train_seq_dir ./train_seq_dir \
  --test_seq_dir ./test_seq_dir \
  --train_npy_dir ./train_dir \
  --test_npy_dir ./val_dir \
  --median 0
```

### 2. NCTC Interval Generation

`data_process/nctc_interval_generate.py` can generate synthetic IPD sequences for IPCTC, TRCTC, Jitterbug, LNCTC, KLIBUR, KLIBUR-O, and other NCTC methods. Partial data is provided in `nctc_intervals/`, and the complete interval data is available at https://huggingface.co/datasets/rulost0220/TRANCTC_NCTC_intervals.

`nctc_intervals/` is intended as a small data example for GitHub users to quickly test the workflow, check input formats, build a small number of test flows, or reproduce part of the experimental pipeline.

Example command:

```bash
python -u data_process/nctc_interval_generate.py \
  --method all \
  --reference_dir ./reference_dir \
  --save_dir ./nctc_interval_dir \
  --ipds_count 20000000 \
  --collect_reference
```

### 3. Detection Flow Construction

After NCTC IPD sequences are generated or prepared, they can be written into the `interval` field of normal test-flow CSV files to build flows under detection.

Main files:

- `data_process/nctc_data_build.py`: writes existing IPD sequences into test-flow CSV files.
- `data_process/get_noise_data.py`: adds Gaussian noise to flows.

Example command:

```bash
python -u data_process/nctc_data_build.py \
  --mode flow \
  --nctc_interval_dir ./nctc_intervals \
  --test_csv_dir ./test_dir \
  --save_dir ./nctc_dir
```

### 4. Model Training

The training entry point is `model/main.py`, and the model definitions are in `model/model.py`.

Example command:

```bash
python -u ./model/main.py \
  --model_type Transformer_packets_feature_model \
  --is_middle True \
  --feature_dim 7 \
  --dim_feedforward 256 \
  --seq_len 8 \
  --d_model 100 \
  --nhead 4 \
  --num_layers 2 \
  --median 1.389503 \
  --batch_size 256 \
  --epochs 30 \
  --dropout 0.1 \
  --lr 0.001 \
  --train_dir ./train_dir \
  --val_dir ./val_dir \
  --save_dir ./checkpoint_dir \
  --gpu_id 0 \
  --masked_idx -1
```

After training, model checkpoints are saved under the directory specified by `save_dir`.

### 5. NCTC Detection

The detection entry point is `detect/nctc_detect.py`. It loads a trained Transformer checkpoint, computes loss scores for reference normal flows and flows under detection, and identifies anomalous flows according to the percentile range of the reference scores.

Example command:

```bash
python -u ./detect/nctc_detect.py \
  --model_type Transformer_packets_feature_model \
  --feature_dim 7 \
  --d_model 100 \
  --nhead 4 \
  --num_layers 2 \
  --dim_feedforward 256 \
  --seq_len 8 \
  --checkpoint_path ./checkpoint_dir/epoch.pth \
  --is_middle True \
  --dropout 0.1 \
  --num_samples 0 \
  --normal_dir ./normal_dir \
  --nctc_root ./nctc_dir \
  --median 1.389503 \
  --batch_size 256 \
  --gpu_id 0 \
  --masked_idx -1 \
  --flow_num 0 \
  --reference_limit 0 \
  --low_q 0 \
  --high_q 99
```

The detection output includes the number of detected flows, the number of flows classified as NCTC or anomalous, and the number of flows classified as normal.

