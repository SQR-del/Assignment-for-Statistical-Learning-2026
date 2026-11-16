#!/bin/bash
# run_all.sh  随机抽药模拟（官方流程），4 种 T/NK 细胞并行，约占 80 个核
# 用法: bash run_all.sh [重复次数=30]
cd "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/2_label_generation" || exit 1
mkdir -p logs; rm -f logs/finished
R=${1:-30}
for ct in "NK cells" "T cells CD4+" "T cells CD8+" "T regulatory cells"; do
  Rscript 01_sim.R "$ct" "$R" 20 > "logs/sim_$(echo -n "$ct" | tr -c '[:alnum:]' '_').log" 2>&1 &
done
wait
echo ALL_DONE > logs/finished
