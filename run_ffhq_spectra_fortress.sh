#!/usr/bin/env bash
set -euo pipefail
cd /home/yanch/GMM
env -u BASH_ENV -u VIRTUALENVWRAPPER_PYTHON \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
STYLEGAN2_USE_CUDA_EXT=0 \
PYTHON=/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python \
PYTHONPATH=/dev/shm/gmm_torch:/home/yanch/GMM \
DATASET=FFHQ DATASET_SLUG=ffhq \
DATA_ROOT=/mnt/sda/rainbow/yanch_gmm/data \
DATA_PATH=/mnt/sda/rainbow/yanch_gmm/data/ffhq_eval \
CACHE_ROOT=/mnt/sda/rainbow/yanch_gmm/cache/ffhq \
MODEL_CHECKPOINT=/mnt/sda/rainbow/yanch_gmm/models/ffhq_resnet18.pth \
STYLEGAN2_CHECKPOINT=/mnt/sda/rainbow/yanch_gmm/models/ffhq_stylegan2/stylegan2-ffhq-config-f.pt \
FFHQ_TRAIN_CACHE=/mnt/sda/rainbow/yanch_gmm/data/ffhq_train64 \
OUT=/mnt/sda/rainbow/yanch_gmm/comparison_results_fortress/ffhq_spectra \
RUN_ID=ffhq_spectra_fortress_all5_seed1234 \
ATTACKS="geiping yin ggl gias gifd" METHODS="spectra" FULL_MATRIX=false \
SEEDS="1234" NUM_EXP=10 TARGET_INDICES="0 1 2 3 4 5 6 7 8 9" \
RESTARTS=1 PIXEL_ITERS=1000 GGL_ITERS=300 GGL_CMA_BUDGET=150 GGL_NUM_SAMPLES=10 \
GAN_ITERS=1000 GIAS_LATENT_ITERS=300 GIAS_PARAM_ITERS=300 GIFD_STAGE_ITERS=50 \
CENSOR_TRIALS=20 ACC_BATCHES=5 GIA_NUM_WORKERS=0 \
SPECTRA_VIRTUAL_LR=0.0001 SPECTRA_BIT_BUDGET=128 SPECTRA_SUBSPACE_DIM=32 \
SPECTRA_FAIL_ON_WEAK_PRIVACY=false SPECTRA_SEED=1234 \
bash /home/yanch/GMM/run_ffhq_five_attack_full_comparison.sh
