#!/bin/bash
# 用法: bash batch.sh <GPU 编号> <名称1> '<JSON1>' <名称2> '<JSON2>' ...   依次在同一块 GPU 上运行
cd "$(dirname "$0")"; g=$1; shift
while [ $# -gt 1 ]; do CUDA_VISIBLE_DEVICES=$g python3 run_variant.py "$1" "$2" > "log_$1.txt" 2>&1; shift 2; done
echo done > "batch_$g.done"
