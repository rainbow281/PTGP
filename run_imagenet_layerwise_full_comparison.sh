#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-ResNet18ImageNet}"
DATA_PATH="${DATA_PATH:-}"
OUT="${OUT:-/tmp/comparison_results/imagenet_layerwise_full}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/gmm_imagenet_cache}"
ATTACKS="${ATTACKS:-geiping yin ggl gias gifd}"
METHODS="${METHODS:-none noise clipping sparsi soteria censor tine_dus}"
SEEDS="${SEEDS:-1234 2234 3234 4234 5234}"
NUM_EXP="${NUM_EXP:-5}"
PIXEL_ITERS="${PIXEL_ITERS:-1000}"
GAN_ITERS="${GAN_ITERS:-1000}"
GIAS_ITERS="${GIAS_ITERS:-8000}"
GGL_CMA_BUDGET="${GGL_CMA_BUDGET:-800}"
GIFD_STAGE_ITERS="${GIFD_STAGE_ITERS:-1000}"
RESTARTS="${RESTARTS:-4}"
ACC_BATCHES="${ACC_BATCHES:-5}"
CENSOR_TRIALS="${CENSOR_TRIALS:-20}"
GMM_CANDIDATES="${GMM_CANDIDATES:-5}"
NOISE_STD="${NOISE_STD:-0.01}"
CLIP_BOUND="${CLIP_BOUND:-4}"
SPARSI_PERCENT="${SPARSI_PERCENT:-20}"
BIGGAN_MODEL="${BIGGAN_MODEL:-biggan-deep-256}"
BIGGAN_CACHE="${BIGGAN_CACHE:-/tmp/models/biggan}"
export GIA_NUM_WORKERS="${GIA_NUM_WORKERS:-0}"
export TORCH_HOME="$CACHE_ROOT/torch"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/runtime"
export PYTORCH_PRETRAINED_BIGGAN_CACHE="$BIGGAN_CACHE"
PRETRAINED_CACHE_SOURCE="${PRETRAINED_CACHE_SOURCE:-${HOME}/.cache/torch/hub/checkpoints}"

case "$OUT" in /*) ;; *) echo "Experiment notice: $OUT" >&2; exit 2;; esac
mkdir -p "$OUT" || { echo "Experiment notice: $OUT" >&2; exit 2; }
[[ -w "$OUT" ]] || { echo "Experiment notice: $OUT" >&2; exit 2; }

is_imagenet_root() {
  local root="$1"
  [[ -d "$root/val" ]] || return 1
  [[ "$(find "$root/val" -mindepth 1 -maxdepth 1 -type d -name 'n????????' | wc -l | tr -d ' ')" == "1000" ]]
}

if [[ -z "$DATA_PATH" ]]; then
  for candidate in \
    /tmp/data/imagenet /tmp/imagenet /tmp/ImageNet \
    /home/yanch/data/imagenet /home/yanch/datasets/imagenet \
    /home/yanch/imagenet /home/yanch/ImageNet \
    /data/imagenet /datasets/imagenet; do
    if is_imagenet_root "$candidate"; then DATA_PATH="$candidate"; break; fi
  done
fi
if [[ -z "$DATA_PATH" ]]; then
  while IFS= read -r val_candidate; do
    candidate="$(dirname "$val_candidate")"
    if is_imagenet_root "$candidate"; then DATA_PATH="$candidate"; break; fi
  done < <(find /tmp /home/yanch -maxdepth 6 -type d -name val 2>/dev/null || true)
fi

if [[ -z "$DATA_PATH" ]] || ! is_imagenet_root "$DATA_PATH"; then
  echo "Experiment notice: " >&2
  echo "Experiment notice: " >&2
  echo "Experiment notice: " >&2
  echo "Experiment notice: $0" >&2
  exit 3
fi
echo "Experiment notice: $DATA_PATH"

if [[ -z "${MODEL_CHECKPOINT:-}" ]] || [[ ! -f "$MODEL_CHECKPOINT" ]]; then
  echo "Experiment notice: " >&2
  exit 4
fi
if [[ -z "${IMAGENET_CLASS_INDEX_JSON:-}" ]] || [[ ! -f "$IMAGENET_CLASS_INDEX_JSON" ]]; then
  echo "Experiment notice: " >&2
  exit 4
fi

mkdir -p "$OUT" "$TORCH_HOME/hub/checkpoints" "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR" "$BIGGAN_CACHE"

for weight_file in vgg16-397923af.pth alexnet-owt-7be5be79.pth; do
  if [[ ! -f "$TORCH_HOME/hub/checkpoints/$weight_file" ]]; then
    if [[ -f "$PRETRAINED_CACHE_SOURCE/$weight_file" ]]; then
      cp "$PRETRAINED_CACHE_SOURCE/$weight_file" "$TORCH_HOME/hub/checkpoints/$weight_file"
    else
      echo "Experiment notice: $weight_file $TORCH_HOME" >&2
      exit 4
    fi
  fi
done

model_args=(--set trained_model=true --set "model_checkpoint_path=$MODEL_CHECKPOINT")

if [[ "${CONFIG_ONLY:-false}" == "true" ]]; then
  echo "ATTACKS=$ATTACKS"
  echo "METHODS=$METHODS"
  echo "SEEDS=$SEEDS"
  echo "NUM_EXP=$NUM_EXP"
  echo "DATA_PATH=$DATA_PATH"
  echo "OUT=$OUT"
  echo "BIGGAN_MODEL=$BIGGAN_MODEL"
  echo "BIGGAN_CACHE=$BIGGAN_CACHE"
  exit 0
fi

for seed in $SEEDS; do
  echo "============================================================"
  echo "Experiment notice: ${seed}"
  for attack in $ATTACKS; do
    echo "Experiment notice: ${seed} ${attack}"
    case "$attack" in
      geiping)
        attack_args=(--set optim=GAN_free --set geiping=true --set yin=false \
          --set ggl=false --set gias=false --set gifd=false \
          --set generative_model= --set max_iterations="$PIXEL_ITERS")
        ;;
      yin)
        attack_args=(--set optim=GAN_free --set geiping=false --set yin=true \
          --set ggl=false --set gias=false --set gifd=false \
          --set generative_model= --set max_iterations="$PIXEL_ITERS")
        ;;
      ggl)
        attack_args=(--set optim=GAN_based --set generative_model=BigGAN \
          --set gen_dataset=ImageNet64 --set biggan_model="$BIGGAN_MODEL" \
          --set biggan_cache_dir="$BIGGAN_CACHE" --set geiping=false --set yin=false \
          --set ggl=true --set gias=false --set gifd=false \
          --set start_layer=0 --set end_layer=13 \
          --set max_iterations="$GAN_ITERS" --set cma_budget="$GGL_CMA_BUDGET" --set num_sample=50)
        ;;
      gias)
        attack_args=(--set optim=GAN_based --set generative_model=BigGAN \
          --set gen_dataset=ImageNet64 --set biggan_model="$BIGGAN_MODEL" \
          --set biggan_cache_dir="$BIGGAN_CACHE" --set geiping=false --set yin=false \
          --set ggl=false --set gias=true --set gifd=false \
          --set start_layer=0 --set end_layer=13 \
          --set max_iterations="$GAN_ITERS" --set gias_iterations="$GIAS_ITERS" --set gias_lr=0.00001)
        ;;
      gifd)
        attack_args=(--set optim=GAN_based --set generative_model=BigGAN \
          --set gen_dataset=ImageNet64 --set biggan_model="$BIGGAN_MODEL" \
          --set biggan_cache_dir="$BIGGAN_CACHE" --set geiping=false --set yin=false \
          --set ggl=false --set gias=false --set gifd=true \
          --set max_iterations="$GAN_ITERS" --set start_layer=0 --set end_layer=13 \
          --set "steps=[$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS]" \
          --set 'lr_io=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]' \
          --set project=true --set do_project_gen_out=true \
          --set do_project_noises=false --set do_project_latent=false \
          --set 'max_radius_gen_out=[1000,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500]' \
          --set 'max_radius_noises=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]' \
          --set 'max_radius_latent=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]')
        ;;
      *) echo "Experiment notice: $attack" >&2; exit 5;;
    esac

    for method in $METHODS; do
      echo "[RUN] dataset=ImageNet64 seed=${seed} attack=${attack} defense=${method}"
      case "$method" in
        tine_dus)
          defense_args=(--set defense_method=tine_dus \
            --set defense_setting.tine_dus.candidate_pool="${TINE_POOL:-12}" \
            --set defense_setting.tine_dus.select_count="${TINE_SELECT:-6}" \
            --set defense_setting.tine_dus.task_rank="${TINE_TASK_RANK:-2}" \
            --set defense_setting.tine_dus.sensitive_rank="${TINE_SENSITIVE_RANK:-4}" \
            --set defense_setting.tine_dus.task_noise_std="${TINE_TASK_NOISE_STD:-0.01}" \
            --set defense_setting.tine_dus.fd_epsilon="${TINE_FD_EPSILON:-0.01}" \
            --set defense_setting.tine_dus.virtual_lr="${TINE_VIRTUAL_LR:-0.0001}" \
            --set defense_setting.tine_dus.utility_beta="${TINE_UTILITY_BETA:-10.0}" \
            --set defense_setting.tine_dus.logdet_reg="${TINE_LOGDET_REG:-0.000001}" \
            --set defense_setting.tine_dus.norm_match=false \
            --set defense_setting.tine_dus.seed="$seed") ;;

        none) defense_args=(--set defense_method=null) ;;
        noise) defense_args=(--set defense_method=noise --set defense_setting.noise="$NOISE_STD") ;;
        clipping) defense_args=(--set defense_method=clipping --set defense_setting.clipping="$CLIP_BOUND") ;;
        sparsi) defense_args=(--set defense_method=compression --set defense_setting.compression="$SPARSI_PERCENT") ;;
        soteria) defense_args=(--set defense_method=representation --set defense_setting.representation=10) ;;
        censor) defense_args=(--set defense_method=orthogonal --set our_num_tries="$CENSOR_TRIALS") ;;

        *) echo "Experiment notice: $method" >&2; exit 6;;
      esac

      "$PYTHON" run_rec.py --config default_config.yml \
        --set exp_name="imagenet_${attack}_${method}_seed${seed}" \
        --set output_dir="$OUT" --set data_path="$DATA_PATH" \
        --set dataset=IMAGENET_IO --set model="$MODEL_NAME" \
        "${attack_args[@]}" "${defense_args[@]}" "${model_args[@]}" \
        --set num_images=1 --set num_exp="$NUM_EXP" --set target_id=0 \
        --set restarts="$RESTARTS" --set train_epochs=1 \
        --set accuracy_eval_batches="$ACC_BATCHES" --set post_attack_train_batches=0 \
        --set set_seed="$seed" --set attack_seed="$seed"
    done
    echo "Experiment notice: ${seed} ${attack}"
  done
done

"$PYTHON" visualize_cifar10_results.py --input-root "$OUT" --output-dir "$OUT/visualization"
