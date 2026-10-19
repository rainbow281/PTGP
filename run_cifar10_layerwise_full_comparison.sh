#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-CIFAR10}"
case "$DATASET" in
  CIFAR10) DATASET_SLUG=cifar10 ;;
  CIFAR100) DATASET_SLUG=cifar100 ;;
  *) echo "Experiment notice: $DATASET" >&2; exit 2 ;;
esac
RESTARTS="${RESTARTS:-4}"
DATA_PATH="${DATA_PATH:-/tmp/data}"
OUT="${OUT:-/tmp/comparison_results/cifar10_layerwise_full}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/spectra_cache}"
ATTACKS="${ATTACKS:-}"
METHODS="${METHODS:-none noise clipping sparsi soteria censor spectra}"
SEEDS="${SEEDS:-1234 2234 3234 4234 5234}"
NUM_EXP="${NUM_EXP:-5}"
TARGET_INDICES="${TARGET_INDICES:-0 1000 2000 3000 4000}"
PIXEL_ITERS="${PIXEL_ITERS:-1000}"
GAN_ITERS="${GAN_ITERS:-1000}"
GIAS_ITERS="${GIAS_ITERS:-8000}"
GGL_ITERS="${GGL_ITERS:-$GAN_ITERS}"
GGL_NUM_SAMPLES="${GGL_NUM_SAMPLES:-50}"
GIAS_LATENT_ITERS="${GIAS_LATENT_ITERS:-$GAN_ITERS}"
GIAS_PARAM_ITERS="${GIAS_PARAM_ITERS:-$GIAS_ITERS}"
GGL_CMA_BUDGET="${GGL_CMA_BUDGET:-800}"
GIFD_STAGE_ITERS="${GIFD_STAGE_ITERS:-1000}"
AUTO_TRAIN_DCGAN="${AUTO_TRAIN_DCGAN:-true}"
DCGAN_EPOCHS="${DCGAN_EPOCHS:-200}"
DCGAN_BATCH_SIZE="${DCGAN_BATCH_SIZE:-128}"
ACC_BATCHES="${ACC_BATCHES:-25}"
CENSOR_TRIALS="${CENSOR_TRIALS:-20}"
GMM_CANDIDATES="${GMM_CANDIDATES:-5}"
NOISE_STD="${NOISE_STD:-0.01}"
CLIP_BOUND="${CLIP_BOUND:-4}"
SPARSI_PERCENT="${SPARSI_PERCENT:-20}"
DCGAN_CHECKPOINT="${DCGAN_CHECKPOINT:-}"
export GIA_NUM_WORKERS="${GIA_NUM_WORKERS:-0}"
export TORCH_HOME="$CACHE_ROOT/torch"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/runtime"
PRETRAINED_CACHE_SOURCE="${PRETRAINED_CACHE_SOURCE:-${HOME}/.cache/torch/hub/checkpoints}"

if [[ -z "$DCGAN_CHECKPOINT" ]]; then
  for checkpoint_candidate in \
    /tmp/models/cifar10_dcgan/netG_epoch_199.pth \
    "$PWD/models/cifar10_dcgan/netG_epoch_199.pth" \
    "$PWD/inversefed/genmodels/cifar10_dcgan/weights/netG_epoch_199.pth"; do
    if [[ -f "$checkpoint_candidate" ]]; then
      DCGAN_CHECKPOINT="$checkpoint_candidate"
      break
    fi
  done
  DCGAN_CHECKPOINT="${DCGAN_CHECKPOINT:-/tmp/models/cifar10_dcgan/netG_epoch_199.pth}"
fi

case " ${ATTACKS:-ggl gias gifd} " in
  *" ggl "*|*" gias "*|*" gifd "*) needs_dcgan=true ;;
  *) needs_dcgan=false ;;
esac
if [[ "$needs_dcgan" == true && ! -f "$DCGAN_CHECKPOINT" && "$AUTO_TRAIN_DCGAN" == true ]]; then
  echo "Experiment notice: ${DCGAN_EPOCHS}"
  "$PYTHON" train_cifar10_dcgan.py \
    --data-path "$DATA_PATH" --output "$DCGAN_CHECKPOINT" \
    --epochs "$DCGAN_EPOCHS" --batch-size "$DCGAN_BATCH_SIZE" \
    --workers "$GIA_NUM_WORKERS"
fi

if [[ -z "$ATTACKS" ]]; then
  if [[ -f "$DCGAN_CHECKPOINT" ]]; then
    ATTACKS="geiping yin ggl gias gifd"
    echo "Experiment notice: $ATTACKS"
  else
    ATTACKS="geiping yin"
    echo "Experiment notice: " >&2
    echo "Experiment notice: $ATTACKS" >&2
    echo "Experiment notice: ${DCGAN_CHECKPOINT}" >&2
  fi
fi

case "$DATA_PATH" in
  /*) ;;
  *) echo "Experiment notice: $DATA_PATH" >&2; exit 2 ;;
esac
case "$OUT" in
  /*) ;;
  *) echo "Experiment notice: $OUT" >&2; exit 2 ;;
esac
mkdir -p "$DATA_PATH" "$OUT" || { echo "Experiment notice: " >&2; exit 2; }
[[ -r "$DATA_PATH" ]] || { echo "Experiment notice: $DATA_PATH" >&2; exit 2; }
[[ -w "$OUT" ]] || { echo "Experiment notice: $OUT" >&2; exit 2; }
read -r -a target_index_array <<< "$TARGET_INDICES"
[[ "${#target_index_array[@]}" -eq "$NUM_EXP" ]] || {
  echo "Experiment notice: $NUM_EXP" >&2; exit 2; }
target_indices_yaml="[$(IFS=,; echo "${target_index_array[*]}")]"

model_args=(--set trained_model=false)
if [[ -n "${MODEL_CHECKPOINT:-}" ]]; then
  model_args=(--set trained_model=true --set "model_checkpoint_path=$MODEL_CHECKPOINT")
fi

mkdir -p "$DATA_PATH" "$OUT" "$TORCH_HOME/hub/checkpoints" "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"

for weight_file in vgg16-397923af.pth alexnet-owt-7be5be79.pth; do
  if [[ ! -f "$TORCH_HOME/hub/checkpoints/$weight_file" ]]; then
    if [[ -f "$PRETRAINED_CACHE_SOURCE/$weight_file" ]]; then
      cp "$PRETRAINED_CACHE_SOURCE/$weight_file" "$TORCH_HOME/hub/checkpoints/$weight_file"
    else
      echo "Experiment notice: $weight_file" >&2
      echo "Experiment notice: $TORCH_HOME $weight_file" >&2
      exit 3
    fi
  fi
done

case " $ATTACKS " in
  *" ggl "*|*" gias "*|*" gifd "*)
    if [[ ! -f "$DCGAN_CHECKPOINT" ]]; then
      echo "Experiment notice: $DCGAN_CHECKPOINT" >&2
      echo "Experiment notice: " >&2
      echo "Experiment notice: " >&2
      exit 4
    fi
    ;;
esac

if [[ "${CONFIG_ONLY:-false}" == "true" ]]; then
  echo "ATTACKS=$ATTACKS"
  echo "METHODS=$METHODS"
  echo "DATA_PATH=$DATA_PATH"
  echo "OUT=$OUT"
  echo "DCGAN_CHECKPOINT=$DCGAN_CHECKPOINT"
  exit 0
fi

for seed in $SEEDS; do
  echo "============================================================"
  echo "Experiment notice: $seed"
  for attack in $ATTACKS; do
    echo "Experiment notice: $seed $attack"
    attack_args=(--set optim=GAN_free --set geiping=false --set yin=false \
      --set ggl=false --set gias=false --set gifd=false \
      --set max_iterations="$PIXEL_ITERS")
    case "$attack" in
      geiping)
        attack_args=(--set optim=GAN_free --set geiping=true --set yin=false \
          --set ggl=false --set gias=false --set gifd=false \
          --set max_iterations="$PIXEL_ITERS")
        ;;
      yin)
        attack_args=(--set optim=GAN_free --set geiping=false --set yin=true \
          --set ggl=false --set gias=false --set gifd=false \
          --set max_iterations="$PIXEL_ITERS")
        ;;
      ggl)
        attack_args=(--set optim=GAN_based --set generative_model=DCGAN \
          --set gen_dataset=C10 --set geiping=false --set yin=false \
          --set ggl=true --set gias=false --set gifd=false \
          --set generator_checkpoint_path="$DCGAN_CHECKPOINT" \
          --set max_iterations="$GGL_ITERS" --set cma_budget="$GGL_CMA_BUDGET" \
          --set num_sample="$GGL_NUM_SAMPLES")
        ;;
      gias)
        attack_args=(--set optim=GAN_based --set generative_model=DCGAN \
          --set gen_dataset=C10 --set geiping=false --set yin=false \
          --set ggl=false --set gias=true --set gifd=false \
          --set generator_checkpoint_path="$DCGAN_CHECKPOINT" \
          --set max_iterations="$GIAS_LATENT_ITERS" --set gias_iterations="$GIAS_PARAM_ITERS" \
          --set gias_lr=0.001)
        ;;
      gifd)
        attack_args=(--set optim=GAN_based --set generative_model=DCGAN \
          --set gen_dataset=C10 --set geiping=false --set yin=false \
          --set ggl=false --set gias=false --set gifd=true \
          --set generator_checkpoint_path="$DCGAN_CHECKPOINT" \
          --set max_iterations="$GAN_ITERS" \
          --set "steps=[$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS]" \
          --set 'lr_io=[0.05,0.05,0.03,0.01]' \
          --set start_layer=0 --set end_layer=3 \
          --set project=true --set do_project_gen_out=true \
          --set do_project_noises=false --set do_project_latent=false \
          --set 'max_radius_gen_out=[1000,2000,3000,4000]' \
          --set 'max_radius_noises=[-1,-1,-1,-1]' \
          --set 'max_radius_latent=[-1,-1,-1,-1]')
        ;;
      *)
        echo "Experiment notice: $attack" >&2
        exit 2
        ;;
    esac

    for method in $METHODS; do
      echo "[RUN] seed=$seed attack=$attack defense=$method"
      defense_args=()
      case "$method" in
        spectra)
          defense_args=(--set defense_method=spectra \
            --set defense_setting.spectra.candidate_pool="${SPECTRA_POOL:-16}" \
            --set defense_setting.spectra.sketch_block="${SPECTRA_SKETCH_BLOCK:-256}" \
            --set defense_setting.spectra.anchor_norm_ratio="${SPECTRA_ANCHOR_NORM_RATIO:-0.35}" \
            --set defense_setting.spectra.anchor_scale_min="${SPECTRA_ANCHOR_SCALE_MIN:-0.25}" \
            --set defense_setting.spectra.anchor_scale_max="${SPECTRA_ANCHOR_SCALE_MAX:-1.00}" \
            --set defense_setting.spectra.synthetic_scale_min="${SPECTRA_SYNTHETIC_SCALE_MIN:-0.35}" \
            --set defense_setting.spectra.synthetic_scale_max="${SPECTRA_SYNTHETIC_SCALE_MAX:-1.50}" \
            --set defense_setting.spectra.virtual_lr="${SPECTRA_VIRTUAL_LR:-0.0001}" \
            --set defense_setting.spectra.max_utility_loss="${SPECTRA_MAX_UTILITY_LOSS:-0.30}" \
            --set defense_setting.spectra.min_relative_change="${SPECTRA_MIN_RELATIVE_CHANGE:-0.85}" \
            --set defense_setting.spectra.max_gradient_cosine="${SPECTRA_MAX_GRADIENT_COSINE:-0.45}" \
            --set defense_setting.spectra.max_residual_cosine="${SPECTRA_MAX_RESIDUAL_COSINE:-0.12}" \
            --set defense_setting.spectra.utility_penalty_weight="${SPECTRA_UTILITY_PENALTY_WEIGHT:-4.0}" \
            --set defense_setting.spectra.privacy_weight_change="${SPECTRA_PRIVACY_WEIGHT_CHANGE:-0.25}" \
            --set defense_setting.spectra.privacy_weight_cosine="${SPECTRA_PRIVACY_WEIGHT_COSINE:-0.30}" \
            --set defense_setting.spectra.privacy_weight_residual="${SPECTRA_PRIVACY_WEIGHT_RESIDUAL:-0.45}" \
            --set defense_setting.spectra.fail_on_weak_privacy="${SPECTRA_FAIL_ON_WEAK_PRIVACY:-false}" \
            --set defense_setting.spectra.probe_count="${SPECTRA_PROBE_COUNT:-4}" \
            --set defense_setting.spectra.probe_noise_std="${SPECTRA_PROBE_NOISE_STD:-0.10}" \
            --set defense_setting.spectra.probe_max_shift="${SPECTRA_PROBE_MAX_SHIFT:-2}" \
            --set defense_setting.spectra.probe_cutout_fraction="${SPECTRA_PROBE_CUTOUT_FRACTION:-0.25}" \
            --set defense_setting.spectra.sage_min_replace="${SPECTRA_SAGE_MIN_REPLACE:-0.55}" \
            --set defense_setting.spectra.sage_max_replace="${SPECTRA_SAGE_MAX_REPLACE:-1.00}" \
            --set defense_setting.spectra.fisher_directions="${SPECTRA_FISHER_DIRECTIONS:-4}" \
            --set defense_setting.spectra.fisher_fd_epsilon="${SPECTRA_FISHER_FD_EPSILON:-0.03}" \
            --set defense_setting.spectra.fisher_rank_weight="${SPECTRA_FISHER_RANK_WEIGHT:-0.75}" \
            --set defense_setting.spectra.fisher_shrinkage="${SPECTRA_FISHER_SHRINKAGE:-0.25}" \
            --set defense_setting.spectra.mira_rd_views="${SPECTRA_MIRA_RD_VIEWS:-4}" \
            --set defense_setting.spectra.mira_rd_weight="${SPECTRA_MIRA_RD_WEIGHT:-1.0}" \
            --set defense_setting.spectra.mira_downsample_factor="${SPECTRA_MIRA_DOWNSAMPLE_FACTOR:-4}" \
            --set defense_setting.spectra.mira_block_size="${SPECTRA_MIRA_BLOCK_SIZE:-4}" \
            --set defense_setting.spectra.mira_min_replace="${SPECTRA_MIRA_MIN_REPLACE:-0.55}" \
            --set defense_setting.spectra.mira_max_replace="${SPECTRA_MIRA_MAX_REPLACE:-1.00}" \
            --set defense_setting.spectra.dro_temperature="${SPECTRA_DRO_TEMPERATURE:-0.35}" \
            --set defense_setting.spectra.dro_weight="${SPECTRA_DRO_WEIGHT:-1.0}" \
            --set defense_setting.spectra.structural_scales="${SPECTRA_STRUCTURAL_SCALES:-2,4,8}" \
            --set defense_setting.spectra.structural_modes="${SPECTRA_STRUCTURAL_MODES:-3}" \
            --set defense_setting.spectra.structural_weight="${SPECTRA_STRUCTURAL_WEIGHT:-0.75}" \
            --set defense_setting.spectra.cosi_signature_bins="${SPECTRA_COSI_SIGNATURE_BINS:-64}" \
            --set defense_setting.spectra.cosi_ridge="${SPECTRA_COSI_RIDGE:-0.001}" \
            --set defense_setting.spectra.landscape_directions="${SPECTRA_LANDSCAPE_DIRECTIONS:-4}" \
            --set defense_setting.spectra.landscape_fd_epsilon="${SPECTRA_LANDSCAPE_FD_EPSILON:-0.03}" \
            --set defense_setting.spectra.landscape_signature_bins="${SPECTRA_LANDSCAPE_SIGNATURE_BINS:-64}" \
            --set defense_setting.spectra.landscape_weight="${SPECTRA_LANDSCAPE_WEIGHT:-1.0}" \
            --set defense_setting.spectra.landscape_selection_weight="${SPECTRA_LANDSCAPE_SELECTION_WEIGHT:-1.0}" \
            --set defense_setting.spectra.global_basins="${SPECTRA_GLOBAL_BASINS:-4}" \
            --set defense_setting.spectra.global_temperature="${SPECTRA_GLOBAL_TEMPERATURE:-0.35}" \
            --set defense_setting.spectra.global_diversity_weight="${SPECTRA_GLOBAL_DIVERSITY_WEIGHT:-0.50}" \
            --set defense_setting.spectra.global_selection_weight="${SPECTRA_GLOBAL_SELECTION_WEIGHT:-1.0}" \
            --set defense_setting.spectra.global_min_distance="${SPECTRA_GLOBAL_MIN_DISTANCE:-0.20}" \
            --set defense_setting.spectra.frtv_enabled="${SPECTRA_FRTV_ENABLED:-true}" \
            --set defense_setting.spectra.frtv_probes="${SPECTRA_FRTV_PROBES:-8}" \
            --set defense_setting.spectra.frtv_probe_steps="${SPECTRA_FRTV_PROBE_STEPS:-24}" \
            --set defense_setting.spectra.frtv_probe_lr="${SPECTRA_FRTV_PROBE_LR:-0.08}" \
            --set defense_setting.spectra.frtv_probe_jitter="${SPECTRA_FRTV_PROBE_JITTER:-2}" \
            --set defense_setting.spectra.frtv_code_dim="${SPECTRA_FRTV_CODE_DIM:-8}" \
            --set defense_setting.spectra.frtv_code_bits="${SPECTRA_FRTV_CODE_BITS:-3}" \
            --set defense_setting.spectra.frtv_code_clip="${SPECTRA_FRTV_CODE_CLIP:-1.0}" \
            --set defense_setting.spectra.frtv_basis_alt="${SPECTRA_FRTV_BASIS_ALT:-true}" \
            --set defense_setting.spectra.frtv_ridge="${SPECTRA_FRTV_RIDGE:-0.05}" \
            --set defense_setting.spectra.frtv_release_clip="${SPECTRA_FRTV_RELEASE_CLIP:-3.0}" \
            --set defense_setting.spectra.frtv_null_entropy="${SPECTRA_FRTV_NULL_ENTROPY:-2.0}" \
            --set defense_setting.spectra.v46_enabled="${SPECTRA_V46_ENABLED:-true}" \
            --set defense_setting.spectra.v46_task_basis_per_probe="${SPECTRA_V46_TASK_BASIS_PER_PROBE:-2}" \
            --set defense_setting.spectra.v46_random_basis="${SPECTRA_V46_RANDOM_BASIS:-16}" \
            --set defense_setting.spectra.v46_coef_bits="${SPECTRA_V46_COEF_BITS:-4}" \
            --set defense_setting.spectra.v46_coef_clip="${SPECTRA_V46_COEF_CLIP:-1.0}" \
            --set defense_setting.spectra.v46_random_weight="${SPECTRA_V46_RANDOM_WEIGHT:-0.35}" \
            --set defense_setting.spectra.v46_trust_ratio="${SPECTRA_V46_TRUST_RATIO:-1.5}" \
            --set defense_setting.spectra.v46_min_descent="${SPECTRA_V46_MIN_DESCENT:-1e-8}" \
            --set defense_setting.spectra.v46_collision_trials="${SPECTRA_V46_COLLISION_TRIALS:-8}" \
            --set defense_setting.spectra.v46_collision_scale="${SPECTRA_V46_COLLISION_SCALE:-0.35}" \
            --set defense_setting.spectra.v47_enabled="${SPECTRA_V47_ENABLED:-true}" \
            --set defense_setting.spectra.v47_bit_budget="${SPECTRA_V47_BIT_BUDGET:-128}" \
            --set defense_setting.spectra.v47_subspace_dim="${SPECTRA_V47_SUBSPACE_DIM:-32}" \
            --set defense_setting.spectra.cpfr_enabled="${SPECTRA_CPFR_ENABLED:-false}" \
            --set defense_setting.spectra.cpfr_candidates="${SPECTRA_CPFR_CANDIDATES:-24}" \
            --set defense_setting.spectra.cpfr_proto_strength_min="${SPECTRA_CPFR_PROTO_STRENGTH_MIN:-0.70}" \
            --set defense_setting.spectra.cpfr_proto_strength_max="${SPECTRA_CPFR_PROTO_STRENGTH_MAX:-1.30}" \
            --set defense_setting.spectra.cpfr_scale_min="${SPECTRA_CPFR_SCALE_MIN:-0.35}" \
            --set defense_setting.spectra.cpfr_scale_max="${SPECTRA_CPFR_SCALE_MAX:-1.80}" \
            --set defense_setting.spectra.cpfr_max_clean_cosine="${SPECTRA_CPFR_MAX_CLEAN_COSINE:-0.35}" \
            --set defense_setting.spectra.cpfr_min_input_distance="${SPECTRA_CPFR_MIN_INPUT_DISTANCE:-0.90}" \
            --set defense_setting.spectra.cpfr_teacher_weight="${SPECTRA_CPFR_TEACHER_WEIGHT:-0.25}" \
            --set defense_setting.spectra.cpfr_entropy_weight="${SPECTRA_CPFR_ENTROPY_WEIGHT:-0.10}" \
            --set defense_setting.spectra.figr_min_replace="${SPECTRA_FIGR_MIN_REPLACE:-0.55}" \
            --set defense_setting.spectra.figr_max_replace="${SPECTRA_FIGR_MAX_REPLACE:-1.00}" \
            --set defense_setting.spectra.igm_sketch_dim="${SPECTRA_IGM_SKETCH_DIM:-32}" \
            --set defense_setting.spectra.igm_sketch_seed="${SPECTRA_IGM_SKETCH_SEED:-271828}" \
            --set defense_setting.spectra.identifiability_weight="${SPECTRA_IDENTIFIABILITY_WEIGHT:-3.0}" \
            --set defense_setting.spectra.match_floor_weight="${SPECTRA_MATCH_FLOOR_WEIGHT:-2.0}" \
            --set defense_setting.spectra.gradient_cosine_weight="${SPECTRA_GRADIENT_COSINE_WEIGHT:-0.25}" \
            --set defense_setting.spectra.seed="$seed") ;;

        none)
          defense_args=(--set defense_method=null)
          ;;
        noise)
          defense_args=(--set defense_method=noise \
            --set defense_setting.noise="$NOISE_STD")
          ;;
        clipping)
          defense_args=(--set defense_method=clipping \
            --set defense_setting.clipping="$CLIP_BOUND")
          ;;
        sparsi)
          defense_args=(--set defense_method=compression \
            --set defense_setting.compression="$SPARSI_PERCENT")
          ;;
        soteria)
          defense_args=(--set defense_method=representation \
            --set defense_setting.representation=10)
          ;;
        censor)
          defense_args=(--set defense_method=orthogonal \
            --set our_num_tries="$CENSOR_TRIALS")
          ;;

        *)
          echo "Experiment notice: $method" >&2
          exit 2
          ;;
      esac

      "$PYTHON" run_rec.py --config default_config.yml \
        --set exp_name="${DATASET_SLUG}_${attack}_${method}_seed${seed}" \
        --set output_dir="$OUT" --set data_path="$DATA_PATH" \
        --set dataset="$DATASET" --set model=ResNet18 \
        "${attack_args[@]}" "${defense_args[@]}" "${model_args[@]}" \
        --set num_images=1 --set num_exp="$NUM_EXP" --set target_id=0 \
        --set "target_indices=$target_indices_yaml" \
        --set restarts="$RESTARTS" --set train_epochs=1 \
        --set accuracy_eval_batches="$ACC_BATCHES" \
        --set post_attack_train_batches=0 --set set_seed="$seed" \
        --set attack_seed="$seed"
    done
    echo "Experiment notice: $seed $attack"
  done
  echo "Experiment notice: $seed"
done

"$PYTHON" visualize_cifar10_results.py \
  --input-root "$OUT" --output-dir "$OUT/visualization"
