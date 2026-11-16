#!/bin/bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python3 run_variant.py rep_ref_s1 '{}' --base-seed 111 > log_rep_ref_s1.txt 2>&1
CUDA_VISIBLE_DEVICES=0 python3 run_variant.py rep_C_s1 '{"K": 32, "epochs": 800, "dropout": 0.2}' --base-seed 111 > log_rep_C_s1.txt 2>&1
CUDA_VISIBLE_DEVICES=0 python3 run_variant.py rep_Csig_s1 '{"K": 32, "epochs": 800, "dropout": 0.2, "sigma": 0.05}' --base-seed 111 > log_rep_Csig_s1.txt 2>&1
CUDA_VISIBLE_DEVICES=0 python3 run_variant.py rep_K32_s1 '{"K": 32}' --base-seed 111 > log_rep_K32_s1.txt 2>&1
echo done > batch3_0.done
