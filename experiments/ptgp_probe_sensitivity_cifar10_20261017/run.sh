#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/yanch/GMM/experiments/ptgp_probe_sensitivity_cifar10_20261017:/home/yanch/GMM
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STYLEGAN2_USE_CUDA_EXT=0
cd /home/yanch/GMM
exec nice -n 19 /mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python /home/yanch/GMM/experiments/ptgp_probe_sensitivity_cifar10_20261017/run_sweep.py >> /mnt/sda/rainbow/yanch_gmm/ptgp_probe_sensitivity_cifar10_gpu1_20261017/logs/screen.log 2>&1
