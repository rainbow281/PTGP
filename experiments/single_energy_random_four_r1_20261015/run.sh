#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export SPECTRA_EXPERIMENT_ROOT=/mnt/sda/rainbow/yanch_gmm/spectra_single_energy_random_four_r1_gpu1_20261015
export PYTHONPATH=/home/yanch/GMM/experiments/single_energy_random_four_r1_20261015:/home/yanch/GMM
export PYTHONUNBUFFERED=1
cd /home/yanch/GMM
exec nice -n 19 /mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python /home/yanch/GMM/experiments/single_energy_random_four_r1_20261015/matrix.py
