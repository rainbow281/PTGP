#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export SPECTRA_EXPERIMENT_ROOT=/mnt/sda/rainbow/yanch_gmm/privacy_ptgp_ffhq_labels_imagenet_ref20_gpu1_20261016
export PYTHONPATH=/home/yanch/GMM/experiments/ptgp_ffhq_labels_imagenet_ref20_20261016:/home/yanch/GMM
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STYLEGAN2_USE_CUDA_EXT=0
cd /home/yanch/GMM
exec nice -n 19 /mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python /home/yanch/GMM/experiments/ptgp_ffhq_labels_imagenet_ref20_20261016/matrix.py
