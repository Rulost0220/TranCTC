# Transformer-based NCTC Detection

This repository keeps only the main project workflow: data preprocessing, synthetic data generation, Transformer training, and Transformer testing. All data paths are placeholders. Replace directories such as `*_dir` with your local paths before running the scripts.

## Experimental Environment

- Ubuntu 20.04
- PyTorch 1.11
- CUDA 11.3

## 1. Data Preprocessing

Main files:

- `data_process/data_preprocess.py`: generates windowed feature data from flow CSV files for training and validation.
- `data_process/ipd_filter.py`: computes the median of IPD differences.
- `data_process/clean_interval.py`: removes rows whose `interval` values are zero or too large.
- `utils.py`: shared data loading, feature construction, and DataLoader utilities.

Example command:

```bash
python -u data_process/data_preprocess.py \
  --train_seq_dir ./train_seq_dir \
  --test_seq_dir ./test_seq_dir \
  --train_npy_dir ./train_dir \
  --test_npy_dir ./val_dir \
  --median 0
```

Arguments:

- `train_seq_dir` / `test_seq_dir`: preprocessed training and validation sequence CSV directories.
- `train_npy_dir` / `test_npy_dir`: output directories for training and validation features.
- `median=0`: computes the IPD-difference median from the training set and reuses it for validation.

## 2. Synthetic Data

Main files:

- `data_process/nctc_interval_generate.py`: generates synthetic IPD sequences for IPCTC, TRCTC, Jitterbug, LNCTC, and KLIBUR, and can write them back into test flow CSV files.
- `data_process/nctc_data_build.py`: overwrites the `interval` column of test flows with existing IPD sequences to build flows under detection.
- `data_process/get_noise_data.py`: adds Gaussian noise to flows under detection.
- `data_process/get_noise_ref_data.py`: adds Gaussian noise to reference normal flows.

Generate synthetic IPDs and write them into test flows:

```bash
python -u data_process/nctc_interval_generate.py \
  --method all \
  --reference_dir ./reference_dir \
  --save_dir ./nctc_interval_dir \
  --test_dir ./test_dir \
  --ipds_count 20000000 \
  --collect_reference \
  --build_flow
```

If IPD files already exist, build flows under detection directly:

```bash
python -u data_process/nctc_data_build.py \
  --mode flow \
  --nctc_interval_dir ./nctc_interval_dir \
  --test_csv_dir ./test_dir \
  --save_dir ./nctc_dir
```

## 3. Method Training

Main files:

- `Transformer/model.py`: Transformer model definitions.
- `Transformer/main.py`: training entry point.
- `Transformer/run_main.sh`: example training script.

Example command:

```bash
bash Transformer/run_main.sh
```

Equivalent core command:

```bash
python -u ./Transformer/main.py \
  --model_type Transformer_packets_feature_model \
  --is_middle True \
  --train_dir ./train_dir \
  --val_dir ./val_dir \
  --save_dir ./checkpoint_dir
```

The trained model checkpoint is saved under `checkpoint_dir`.

## 4. Testing

Main files:

- `data_process/nctc_detect.py`: loads a Transformer checkpoint, computes scores for reference normal flows and flows under detection, and prints positive/negative statistics.
- `data_process/run_detect.sh`: example detection script.

Example command:

```bash
bash data_process/run_detect.sh
```

Equivalent core command:

```bash
python -u ./data_process/nctc_detect.py \
  --model_type Transformer_packets_feature_model \
  --checkpoint_path ./checkpoint_dir/epoch.pth \
  --normal_dir ./normal_dir \
  --nctc_root ./nctc_dir \
  --is_middle True 
```

Output fields:

- `total`: number of CSV files used for detection.
- `positive`: number of flows classified as anomalous or NCTC.
- `negative`: number of flows classified as normal.
