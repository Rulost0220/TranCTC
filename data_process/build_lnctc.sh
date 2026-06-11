#!/bin/bash
for subdir in 100_200 200_300 300_400 400_500 500_600 600_700 700_800 800_900 900_1000 1000_1100 1100_1200 1200_1300 1300_1400 1400_1500 1500_1600 1600_1700 1700_1800 1800_1900 1900_2000 2000_plus; do
    nohup python -u data_process/lnctc_build.py --subdir $subdir > ./log_dir/log_$subdir 2>&1 &
done
