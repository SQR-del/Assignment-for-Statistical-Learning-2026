#!/bin/bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python3 run_variant.py v2_base '{"K": 32, "epochs": 800, "dropout": 0.2}' --variant base > log_v2_base.txt 2>&1
CUDA_VISIBLE_DEVICES=0 python3 run_export.py base > log_exp_base.txt 2>&1
CUDA_VISIBLE_DEVICES=0 python3 run_export.py base_I2 > log_exp_base_I2.txt 2>&1
echo done > batch4_0.done
