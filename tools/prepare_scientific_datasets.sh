#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/tmp/data}"
DATASETS="${DATASETS:-cifar10}"
mkdir -p "$DATA_ROOT"

contains_dataset() {
  case " $DATASETS " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

if (contains_dataset imagenet || contains_dataset ffhq) && \
   ! "$PYTHON" -c 'import datasets, gdown, requests, PIL' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --only-binary=:all: \
    'numpy==1.26.4' 'pyarrow==17.0.0' 'datasets==2.21.0' \
    'huggingface_hub>=0.23,<1.0' pillow gdown requests
fi

if (contains_dataset cifar10 || contains_dataset cifar100) && \
   ! "$PYTHON" -c 'import torchvision' >/dev/null 2>&1; then
  echo "Experiment notice: $PYTHON" >&2
  echo "Experiment notice: " >&2
  exit 2
fi

if contains_dataset imagenet; then
  "$PYTHON" prepare_full_imagenet.py \
    --output "$DATA_ROOT/imagenet" --splits validation
fi

if contains_dataset ffhq; then
  "$PYTHON" prepare_ffhq_eval_subset.py --output "$DATA_ROOT/ffhq_eval" \
    --indices 0 1000 2000 3000 4000 5000 6000 7000 8000 9000
fi

DATASETS="$DATASETS" DATA_ROOT="$DATA_ROOT" "$PYTHON" - <<'PY'
import os
from torchvision.datasets import CIFAR10, CIFAR100
root = os.environ['DATA_ROOT']
requested = set(os.environ['DATASETS'].lower().split())
classes = []
if 'cifar10' in requested:
    classes.append(CIFAR10)
if 'cifar100' in requested:
    classes.append(CIFAR100)
for cls in classes:
    cls(root=root, train=True, download=True)
    cls(root=root, train=False, download=True)
    print(f'{cls.__name__} is ready.')
PY
