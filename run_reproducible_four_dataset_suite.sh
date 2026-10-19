#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-/home/yanch/GMM/comparison_results}"
DATA_ROOT="${DATA_ROOT:-/mnt/sda/rainbow/yanch_gmm/data}"
DATASETS="${DATASETS:-imagenet cifar10 cifar100 ffhq}"
ATTACKS="${ATTACKS:-geiping yin ggl gias gifd}"
METHODS="${METHODS:-none noise clipping sparsi soteria censor spectra}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/sda/rainbow/yanch_gmm/models}"
BIGGAN_MODEL="${BIGGAN_MODEL:-$MODEL_ROOT/biggan-deep-256}"
AUTO_PREPARE_BIGGAN="${AUTO_PREPARE_BIGGAN:-true}"
export PYTHON OUT DATA_ROOT DATASETS

run_requested() {
  case " $DATASETS " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

if run_requested imagenet; then
  DATASETS=imagenet bash prepare_scientific_datasets.sh
  if [[ "$AUTO_PREPARE_BIGGAN" == true ]]; then
    "$PYTHON" prepare_biggan.py --output "$BIGGAN_MODEL" \
      --search-root /tmp/models --search-root "$HOME/.cache" \
      --search-root /mnt/sda/rainbow
  fi
  REQUIRE_TRAIN=false DOWNLOAD_SPLITS=validation \
  DATA_PATH="$DATA_ROOT/imagenet" RUN_ID=imagenet_five_attack_full \
  ATTACKS="$ATTACKS" METHODS="$METHODS" \
  MODEL_CHECKPOINT="${IMAGENET_MODEL_CHECKPOINT:-$MODEL_ROOT/imagenet_resnet18.pth}" \
  BIGGAN_MODEL="$BIGGAN_MODEL" \
  IMAGENET_CLASS_INDEX_JSON="${IMAGENET_CLASS_INDEX_JSON:-$MODEL_ROOT/imagenet_class_index.json}" \
  bash run_imagenet1k_scientific_comparison.sh
fi

if run_requested cifar10; then
  DATASETS=cifar10 bash prepare_scientific_datasets.sh
  DATASET=CIFAR10 DATA_PATH="$DATA_ROOT" OUT="$OUT/cifar10_five_attack_full" \
  ATTACKS="$ATTACKS" METHODS="$METHODS" \
  MODEL_CHECKPOINT="${CIFAR10_MODEL_CHECKPOINT:-$MODEL_ROOT/cifar10_resnet18.pth}" \
  DCGAN_CHECKPOINT="${CIFAR10_DCGAN_CHECKPOINT:-$MODEL_ROOT/cifar10_dcgan/netG_epoch_199.pth}" \
  bash run_cifar10_layerwise_full_comparison.sh
fi

if run_requested cifar100; then
  DATASETS=cifar100 bash prepare_scientific_datasets.sh
  DATASET=CIFAR100 DATA_PATH="$DATA_ROOT" OUT="$OUT/cifar100_five_attack_full" \
  ATTACKS="$ATTACKS" METHODS="$METHODS" \
  MODEL_CHECKPOINT="${CIFAR100_MODEL_CHECKPOINT:-$MODEL_ROOT/cifar100_resnet18.pth}" \
  DCGAN_CHECKPOINT="${CIFAR100_DCGAN_CHECKPOINT:-$MODEL_ROOT/cifar100_dcgan/netG_epoch_199.pth}" \
  bash run_cifar10_layerwise_full_comparison.sh
fi

if run_requested ffhq; then
  DATASETS=ffhq bash prepare_scientific_datasets.sh
  DATA_PATH="$DATA_ROOT/ffhq_eval" OUT="$OUT/ffhq_five_attack_full" \
  ATTACKS="$ATTACKS" METHODS="$METHODS" \
  MODEL_CHECKPOINT="${FFHQ_MODEL_CHECKPOINT:-$MODEL_ROOT/ffhq_resnet18.pth}" \
  STYLEGAN2_CHECKPOINT="${FFHQ_STYLEGAN2_CHECKPOINT:-$MODEL_ROOT/ffhq_stylegan2/stylegan2-ffhq-config-f.pt}" \
  FFHQ_TRAIN_CACHE="${FFHQ_TRAIN_CACHE:-$DATA_ROOT/ffhq_train64}" \
  bash run_ffhq_five_attack_full_comparison.sh
fi

echo "Requested datasets completed: $DATASETS"
echo "All results: $OUT"
