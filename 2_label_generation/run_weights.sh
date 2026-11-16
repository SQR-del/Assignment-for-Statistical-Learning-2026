#!/bin/bash
# run_weights.sh  权重干预实验，4 种 T/NK 细胞并行。用法: bash run_weights.sh [重复次数=10]
cd "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/2_label_generation" || exit 1
mkdir -p logs; rm -f logs/weights_finished
for ct in "NK cells" "T cells CD4+" "T cells CD8+" "T regulatory cells"; do
  Rscript 07_weights_experiment.R "$ct" "${1:-10}" 16 > "logs/weights_$(echo -n "$ct" | tr -c '[:alnum:]' '_').log" 2>&1 &
done
wait; echo ALL_DONE > logs/weights_finished
