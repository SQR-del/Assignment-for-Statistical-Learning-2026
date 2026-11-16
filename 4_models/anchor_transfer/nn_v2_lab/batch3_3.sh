#!/bin/bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=3 python3 run_aligned.py ref '{}' --mode small > log_al_small_ref.txt 2>&1
CUDA_VISIBLE_DEVICES=3 python3 run_aligned.py ref '{}' --mode calib > log_al_calib_ref.txt 2>&1
CUDA_VISIBLE_DEVICES=3 python3 run_blend.py ref '{}' > log_blend_ref.txt 2>&1
CUDA_VISIBLE_DEVICES=3 python3 run_aligned.py C '{"K": 32, "epochs": 800, "dropout": 0.2}' --mode small > log_al_small_C.txt 2>&1
CUDA_VISIBLE_DEVICES=3 python3 run_aligned.py C '{"K": 32, "epochs": 800, "dropout": 0.2}' --mode calib > log_al_calib_C.txt 2>&1
CUDA_VISIBLE_DEVICES=3 python3 run_blend.py C '{"K": 32, "epochs": 800, "dropout": 0.2}' > log_blend_C.txt 2>&1
echo done > batch3_3.done
