#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export SPECTRA_EXPERIMENT_ROOT=/mnt/sda/rainbow/yanch_gmm/convergence_cifar10_all_methods_censor_protocol_gpu1_20261016
export PYTHONPATH=/home/yanch/GMM/experiments/cifar10_convergence_all_methods_20261016:/home/yanch/GMM
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/yanch/GMM
echo "WAIT_PRIVACY_LOCK $(date -Is)"
exec 9>/mnt/sda/rainbow/yanch_gmm/gpu1_privacy_convergence_20261016.lock
flock -x 9
while [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do
  echo "WAIT_GPU1_OTHER_USER $(date -Is)"; sleep 60
done
exec nice -n 19 /mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python \
  /home/yanch/GMM/experiments/cifar10_convergence_all_methods_20261016/run_all.py
