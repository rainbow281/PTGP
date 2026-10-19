#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL_CHECKPOINT:-}" || ! -f "$MODEL_CHECKPOINT" ]]; then
  echo "Experiment notice: " >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-/tmp/data}"
OUT="${OUT:-/tmp/comparison_results/cifar10_layerwise_formal}"
SEEDS="${SEEDS:-1234 2234 3234 4234 5234}"
NUM_EXP="${NUM_EXP:-5}"
ATTACK_ITERS="${ATTACK_ITERS:-300}"

for method in censor tine_dus gmm_budget; do
  PYTHON="$PYTHON" DATA_PATH="$DATA_PATH" OUT="$OUT" \
  SEEDS="$SEEDS" NUM_EXP="$NUM_EXP" ATTACK_ITERS="$ATTACK_ITERS" \
  ACC_BATCHES=null CENSOR_TRIALS=20 BICONE_CANDIDATES=16 \
  ATTACKS="geiping yin" METHODS="$method" \
  MODEL_CHECKPOINT="$MODEL_CHECKPOINT" \
  ./run_gmm_bicone_repeated.sh
done
