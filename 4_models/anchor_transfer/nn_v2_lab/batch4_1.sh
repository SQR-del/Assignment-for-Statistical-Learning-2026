#!/bin/bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=1 python3 run_variant.py v2_base_I1 '{"K": 32, "epochs": 800, "dropout": 0.2}' --variant base_I1 > log_v2_base_I1.txt 2>&1
CUDA_VISIBLE_DEVICES=1 python3 run_export.py base_I1 > log_exp_base_I1.txt 2>&1
CUDA_VISIBLE_DEVICES=1 python3 run_export.py blend > log_exp_blend.txt 2>&1
echo done > batch4_1.done
