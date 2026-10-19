#!/usr/bin/env bash
set -euo pipefail
cd /home/yanch/GMM

env -u BASH_ENV -u VIRTUALENVWRAPPER_PYTHON \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHON="${PYTHON:-/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python}" \
PYTHONPATH="/dev/shm/gmm_torch:/home/yanch/GMM" \
DATASET=CIFAR100 \
DATASET_SLUG=cifar100 \
DATA_ROOT="${DATA_ROOT:-/mnt/sda/rainbow/yanch_gmm/data}" \
DATA_PATH="${DATA_PATH:-/mnt/sda/rainbow/yanch_gmm/data}" \
CACHE_ROOT="${CACHE_ROOT:-/mnt/sda/rainbow/yanch_gmm/cache/cifar100}" \
PRETRAINED_CACHE_SOURCE="${PRETRAINED_CACHE_SOURCE:-/home/yanch/.cache/torch/hub/checkpoints}" \
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/mnt/sda/rainbow/yanch_gmm/models/cifar100_resnet18.pth}" \
DCGAN_CHECKPOINT="${DCGAN_CHECKPOINT:-/mnt/sda/rainbow/yanch_gmm/models/cifar100_dcgan/netG_epoch_199.pth}" \
AUTO_TRAIN_DCGAN=false \
OUT="${OUT:-/mnt/sda/rainbow/yanch_gmm/comparison_results_geco_fp/cifar100_spectra}" \
RUN_ID="${RUN_ID:-cifar100_spectra_geco_fp_all5_seed1234}" \
ATTACKS="${ATTACKS:-geiping yin ggl gias gifd}" \
METHODS="${METHODS:-spectra}" \
FULL_MATRIX=false \
SEEDS="${SEEDS:-1234}" \
NUM_EXP="${NUM_EXP:-10}" \
TARGET_INDICES="${TARGET_INDICES:-0 1000 2000 3000 4000 5000 6000 7000 8000 9000}" \
RESTARTS="${RESTARTS:-1}" \
PIXEL_ITERS="${PIXEL_ITERS:-1000}" \
GGL_ITERS="${GGL_ITERS:-300}" \
GGL_CMA_BUDGET="${GGL_CMA_BUDGET:-150}" \
GGL_NUM_SAMPLES="${GGL_NUM_SAMPLES:-10}" \
GAN_ITERS="${GAN_ITERS:-1000}" \
GIAS_LATENT_ITERS="${GIAS_LATENT_ITERS:-300}" \
GIAS_PARAM_ITERS="${GIAS_PARAM_ITERS:-300}" \
GIFD_STAGE_ITERS="${GIFD_STAGE_ITERS:-50}" \
CENSOR_TRIALS="${CENSOR_TRIALS:-20}" \
ACC_BATCHES="${ACC_BATCHES:-5}" \
GIA_NUM_WORKERS="${GIA_NUM_WORKERS:-0}" \
SPECTRA_VIRTUAL_LR="${SPECTRA_VIRTUAL_LR:-0.0001}" \
SPECTRA_BIT_BUDGET="${SPECTRA_BIT_BUDGET:-128}" \
SPECTRA_SUBSPACE_DIM="${SPECTRA_SUBSPACE_DIM:-32}" \
SPECTRA_FAIL_ON_WEAK_PRIVACY=false \
bash /home/yanch/GMM/run_cifar10_layerwise_full_comparison.sh
