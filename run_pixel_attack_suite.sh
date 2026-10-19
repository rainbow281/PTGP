#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
DATASET="${DATASET:?Set DATASET to CIFAR10, CIFAR100, or FFHQ64}"
DATA_PATH="${DATA_PATH:-/tmp/data}"
OUT="${OUT:-/home/yanch/GMM/comparison_results}"
METHODS="${METHODS:-none noise clipping sparsi soteria censor tine_dus}"
ATTACKS="${ATTACKS:-geiping yin}"
SEEDS="${SEEDS:-1234}"
RESTARTS="${RESTARTS:-4}"
PIXEL_ITERS="${PIXEL_ITERS:-15000}"

case "$DATASET" in
  CIFAR10) slug=cifar10; indices='[0,1000,2000,3000,4000,5000,6000,7000,8000,9000]' ;;
  CIFAR100) slug=cifar100; indices='[0,1000,2000,3000,4000,5000,6000,7000,8000,9000]' ;;
  FFHQ64) slug=ffhq; indices='[0,1,2,3,4,5,6,7,8,9]' ;;
  *) echo "Unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac
RESULTS="$OUT/reproducible_suite/results"
DONE="$OUT/reproducible_suite/completed"
mkdir -p "$RESULTS" "$DONE"

for seed in $SEEDS; do
  for attack in $ATTACKS; do
    if [[ "$attack" == geiping ]]; then
      attack_args=(--set geiping=true --set yin=false)
    elif [[ "$attack" == yin ]]; then
      attack_args=(--set geiping=false --set yin=true)
    else
      echo "Pixel suite supports geiping and yin only" >&2; exit 3
    fi
    for method in $METHODS; do
      marker="$DONE/${slug}_${attack}_${method}_seed${seed}.done"
      [[ -s "$marker" ]] && grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' "$marker" && { echo "[SKIP] $marker"; continue; }
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
        noise) defense_args=(--set defense_method=noise --set defense_setting.noise=0.1) ;;
        clipping) defense_args=(--set defense_method=clipping --set defense_setting.clipping=4) ;;
        sparsi) defense_args=(--set defense_method=compression --set defense_setting.compression=20) ;;
        soteria) defense_args=(--set defense_method=representation --set defense_setting.representation=10) ;;
        censor) defense_args=(--set defense_method=orthogonal --set our_num_tries=20) ;;

        *) echo "Unsupported defense=$method" >&2; exit 4 ;;
      esac
      "$PYTHON" run_rec.py --config default_config.yml \
        --set exp_name="${slug}_${attack}_${method}_seed${seed}" \
        --set output_dir="$RESULTS" --set data_path="$DATA_PATH" \
        --set dataset="$DATASET" --set model=ResNet18 --set optim=GAN_free \
        "${attack_args[@]}" "${defense_args[@]}" \
        --set cost_fn=sim_cmpr0 --set num_images=1 --set num_exp=10 \
        --set target_id=0 --set "target_indices=$indices" \
        --set restarts="$RESTARTS" --set max_iterations="$PIXEL_ITERS" \
        --set train_epochs=1 --set post_attack_train_batches=0 \
        --set set_seed="$seed" --set attack_seed="$seed"
      date -u +'%Y-%m-%dT%H:%M:%SZ' > "$marker"
    done
  done
done
