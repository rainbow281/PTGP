#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/sda/rainbow/yanch_gmm/gifd_eot_cifar100_ffhq_gpu1_20261018
CODE=/home/yanch/GMM/experiments/gifd_eot_cifar100_ffhq_20261018
PYTHON=/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python
mkdir -p "$ROOT/logs"
export CUDA_VISIBLE_DEVICES=1
export SPECTRA_EXPERIMENT_ROOT="$ROOT"
export PYTHONPATH="$CODE:/home/yanch/GMM"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STYLEGAN2_USE_CUDA_EXT=0

printf '{"phase":"WAITING_GPU1","reason":"existing GPU1 training","gpu":1}\n' > "$ROOT/stage.json"
idle=0
while (( idle < 3 )); do
  reading=$(nvidia-smi --id=1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  IFS=, read -r memory utilization <<< "$reading"
  memory=${memory//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  if [[ "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] &&
     (( memory < 3000 && utilization < 20 )); then
    idle=$((idle + 1))
  else
    idle=0
  fi
  printf '%s GPU1 memory=%sMiB utilization=%s%% idle_checks=%s/3\n' \
    "$(date -Is)" "$memory" "$utilization" "$idle"
  if (( idle < 3 )); then sleep 60; fi
done

cd /home/yanch/GMM
exec nice -n 19 "$PYTHON" "$CODE/run_matrix.py"
