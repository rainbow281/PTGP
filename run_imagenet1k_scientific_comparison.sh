#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-/tmp/data/imagenet}"
OUT="${OUT:-/home/yanch/GMM/comparison_results}"
RUN_ID="${RUN_ID:-censor_protocol_b1_r4_n10}"
RESULTS_DIR="$OUT/$RUN_ID/results"
META_DIR="$OUT/$RUN_ID/metadata"
DONE_DIR="$OUT/$RUN_ID/completed"

ATTACKS="${ATTACKS:-geiping yin ggl gias gifd}"
METHODS="${METHODS:-none noise clipping sparsi soteria censor spectra}"
FULL_MATRIX="${FULL_MATRIX:-true}"
if [[ "$FULL_MATRIX" == true ]]; then
  [[ "$ATTACKS" == "geiping yin ggl gias gifd" ]] || { echo "Experiment notice: " >&2; exit 2; }
  [[ "$METHODS" == "none noise clipping sparsi soteria censor spectra" ]] || { echo "Experiment notice: " >&2; exit 2; }
fi
SEEDS="${SEEDS:-1234}"
TARGET_INDICES="${TARGET_INDICES:-0 5000 10000 15000 20000 25000 30000 35000 40000 45000}"
NUM_EXP="${NUM_EXP:-10}"
RESTARTS="${RESTARTS:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
STRICT_IMAGENET="${STRICT_IMAGENET:-true}"
REQUIRE_TRAIN="${REQUIRE_TRAIN:-false}"
AUTO_DOWNLOAD_IMAGENET="${AUTO_DOWNLOAD_IMAGENET:-true}"
DOWNLOAD_SPLITS="${DOWNLOAD_SPLITS:-validation}"
AUTO_INSTALL_DATASETS="${AUTO_INSTALL_DATASETS:-true}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

PIXEL_ITERS="${PIXEL_ITERS:-1000}"
GGL_ITERS="${GGL_ITERS:-300}"
GGL_CMA_BUDGET="${GGL_CMA_BUDGET:-150}"
GGL_NUM_SAMPLES="${GGL_NUM_SAMPLES:-10}"
GIAS_LATENT_ITERS="${GIAS_LATENT_ITERS:-300}"
GIAS_PARAM_ITERS="${GIAS_PARAM_ITERS:-300}"
GIFD_STAGE_ITERS="${GIFD_STAGE_ITERS:-50}"
CENSOR_TRIALS="${CENSOR_TRIALS:-20}"
GMM_CANDIDATES="${GMM_CANDIDATES:-5}"
NOISE_STD="${NOISE_STD:-0.1}"
CLIP_BOUND="${CLIP_BOUND:-4}"
SPARSI_PERCENT="${SPARSI_PERCENT:-20}"
ACC_BATCHES="${ACC_BATCHES:-1}"

BIGGAN_MODEL="${BIGGAN_MODEL:-biggan-deep-256}"
BIGGAN_CACHE="${BIGGAN_CACHE:-$OUT/cache/biggan}"
CACHE_ROOT="${CACHE_ROOT:-$OUT/cache/runtime}"
IMAGENET_CLASS_INDEX_JSON="${IMAGENET_CLASS_INDEX_JSON:-}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"

mkdir -p "$DATA_PATH" "$RESULTS_DIR" "$META_DIR" "$DONE_DIR" "$CACHE_ROOT/tmp" "$BIGGAN_CACHE"
export GIA_NUM_WORKERS="${GIA_NUM_WORKERS:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-1234}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TORCH_HOME="$CACHE_ROOT/torch"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTORCH_PRETRAINED_BIGGAN_CACHE="$BIGGAN_CACHE"
export GIA_IMAGENET_SIZE="$IMAGE_SIZE"
mkdir -p "$TORCH_HOME/hub/checkpoints" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"

read -r -a target_index_array <<< "$TARGET_INDICES"
if [[ "${#target_index_array[@]}" -ne "$NUM_EXP" ]]; then
  echo "NUM_EXP=$NUM_EXP, but TARGET_INDICES contains ${#target_index_array[@]} indices" >&2
  exit 2
fi
target_indices_yaml="[$(IFS=,; echo "${target_index_array[*]}")]"

need_download=false
val_count=0
train_count=0
[[ -d "$DATA_PATH/val" ]] && val_count="$(find "$DATA_PATH/val" -type f \( -iname '*.JPEG' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
[[ -d "$DATA_PATH/train" ]] && train_count="$(find "$DATA_PATH/train" -type f \( -iname '*.JPEG' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
[[ "$val_count" -ne 50000 ]] && need_download=true
[[ "$REQUIRE_TRAIN" == true && "$train_count" -ne 1281167 ]] && need_download=true
if [[ "$need_download" == true ]]; then
  if [[ "$AUTO_DOWNLOAD_IMAGENET" != true ]]; then
    echo "ImageNet-1K is incomplete under $DATA_PATH and automatic download is disabled." >&2
    exit 2
  fi
  available_kb="$(df -Pk "$DATA_PATH" | awk 'NR==2 {print $4}')"
  required_kb="$((MIN_FREE_GB * 1024 * 1024))"
  if (( available_kb < required_kb )); then
    echo "Insufficient free space for ImageNet-1K under $DATA_PATH." >&2
    echo "Required at least ${MIN_FREE_GB} GiB; available $((available_kb / 1024 / 1024)) GiB." >&2
    exit 2
  fi
  if ! "$PYTHON" -c 'import datasets, PIL' >/dev/null 2>&1; then
    if [[ "$AUTO_INSTALL_DATASETS" != true ]]; then
      echo "Missing Python packages: datasets and/or pillow." >&2
      exit 2
    fi
    "$PYTHON" -m pip install --only-binary=:all: \
      'numpy==1.26.4' 'pyarrow==17.0.0' 'datasets==2.21.0' \
      'huggingface_hub>=0.23,<1.0' pillow
  fi
  read -r -a download_split_array <<< "$DOWNLOAD_SPLITS"
  "$PYTHON" prepare_full_imagenet.py --output "$DATA_PATH" --splits "${download_split_array[@]}"
fi

validator_args=(--data-path "$DATA_PATH" --output "$META_DIR/dataset_manifest.json" --indices "${target_index_array[@]}")
[[ "$STRICT_IMAGENET" == true ]] && validator_args+=(--strict)
[[ "$REQUIRE_TRAIN" == true ]] && validator_args+=(--require-train)
if [[ -n "$IMAGENET_CLASS_INDEX_JSON" ]]; then
  validator_args+=(--class-index-json "$IMAGENET_CLASS_INDEX_JSON")
elif [[ "$ATTACKS" == *ggl* || "$ATTACKS" == *gias* || "$ATTACKS" == *gifd* ]]; then
  echo "WARNING: GAN attacks requested without IMAGENET_CLASS_INDEX_JSON; synset/BigGAN order is not independently verified." >&2
fi
"$PYTHON" validate_imagenet1k.py "${validator_args[@]}"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)'

{
  echo "RUN_ID=$RUN_ID"
  echo "DATA_PATH=$DATA_PATH"
  echo "ATTACKS=$ATTACKS"
  echo "METHODS=$METHODS"
  echo "SEEDS=$SEEDS"
  echo "TARGET_INDICES=$TARGET_INDICES"
  echo "RESTARTS=$RESTARTS"
  echo "IMAGE_SIZE=$IMAGE_SIZE"
  echo "PIXEL_ITERS=$PIXEL_ITERS"
  echo "GGL_CMA_BUDGET=$GGL_CMA_BUDGET"
  echo "GIAS_PARAM_ITERS=$GIAS_PARAM_ITERS"
  echo "GIFD_STAGE_ITERS=$GIFD_STAGE_ITERS"
  echo "CENSOR_TRIALS=$CENSOR_TRIALS"
  "$PYTHON" --version
  "$PYTHON" -c 'import torch, torchvision; print("torch=" + torch.__version__); print("torchvision=" + torchvision.__version__)'
} > "$META_DIR/protocol.txt"

model_args=(--set trained_model=false)
if [[ -n "$MODEL_CHECKPOINT" ]]; then
  model_args=(--set trained_model=true --set "model_checkpoint_path=$MODEL_CHECKPOINT")
fi
if [[ ! -f "$MODEL_CHECKPOINT" ]]; then
  echo "Experiment notice: $MODEL_CHECKPOINT" >&2
  echo "Experiment notice: " >&2
  exit 4
fi
case " $ATTACKS " in
  *" ggl "*|*" gias "*|*" gifd "*)
    [[ -d "$BIGGAN_MODEL" ]] || { echo "Experiment notice: $BIGGAN_MODEL" >&2; exit 4; }
    [[ -f "$IMAGENET_CLASS_INDEX_JSON" ]] || { echo "Experiment notice: $IMAGENET_CLASS_INDEX_JSON" >&2; exit 4; }
    ;;
esac

for seed in $SEEDS; do
  for attack in $ATTACKS; do
    case "$attack" in
      geiping)
        attack_args=(--set optim=GAN_free --set geiping=true --set yin=false --set ggl=false --set gias=false --set gifd=false --set generative_model= --set cost_fn=sim_cmpr0 --set max_iterations="$PIXEL_ITERS")
        ;;
      yin)
        attack_args=(--set optim=GAN_free --set geiping=false --set yin=true --set ggl=false --set gias=false --set gifd=false --set generative_model= --set cost_fn=sim_cmpr0 --set max_iterations="$PIXEL_ITERS")
        ;;
      ggl)
        attack_args=(--set optim=GAN_based --set generative_model=BigGAN --set gen_dataset=ImageNet64 --set biggan_model="$BIGGAN_MODEL" --set biggan_cache_dir="$BIGGAN_CACHE" --set geiping=false --set yin=false --set ggl=true --set gias=false --set gifd=false --set cost_fn=sim_cmpr0 --set start_layer=0 --set end_layer=13 --set max_iterations="$GGL_ITERS" --set cma_budget="$GGL_CMA_BUDGET" --set num_sample="$GGL_NUM_SAMPLES" --set KLD=0.1)
        ;;
      gias)
        attack_args=(--set optim=GAN_based --set generative_model=BigGAN --set gen_dataset=ImageNet64 --set biggan_model="$BIGGAN_MODEL" --set biggan_cache_dir="$BIGGAN_CACHE" --set geiping=false --set yin=false --set ggl=false --set gias=true --set gifd=false --set cost_fn=sim_cmpr0 --set start_layer=0 --set end_layer=13 --set max_iterations="$GIAS_LATENT_ITERS" --set gias_iterations="$GIAS_PARAM_ITERS" --set gias_lr=0.00001)
        ;;
      gifd)
        gifd_steps="[$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS]"
        attack_args=(--set optim=GAN_based --set generative_model=BigGAN --set gen_dataset=ImageNet64 --set biggan_model="$BIGGAN_MODEL" --set biggan_cache_dir="$BIGGAN_CACHE" --set geiping=false --set yin=false --set ggl=false --set gias=false --set gifd=true --set cost_fn=sim_cmpr0 --set max_iterations="$GIAS_LATENT_ITERS" --set start_layer=0 --set end_layer=13 --set "steps=$gifd_steps" --set 'lr_io=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]' --set project=true --set do_project_gen_out=true --set do_project_noises=false --set do_project_latent=false --set 'max_radius_gen_out=[1000,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500]' --set 'max_radius_noises=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]' --set 'max_radius_latent=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]')
        ;;
      *) echo "Unsupported attack: $attack" >&2; exit 3 ;;
    esac

    for method in $METHODS; do
      marker="$DONE_DIR/v2_seed${seed}_${attack}_${method}.done"
      if [[ -s "$marker" ]] && grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' "$marker"; then
        echo "[SKIP completed] seed=$seed attack=$attack defense=$method"
        continue
      fi
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
            --set defense_setting.spectra.figr_min_replace="${SPECTRA_FIGR_MIN_REPLACE:-0.55}" \
            --set defense_setting.spectra.figr_max_replace="${SPECTRA_FIGR_MAX_REPLACE:-1.00}" \
            --set defense_setting.spectra.igm_sketch_dim="${SPECTRA_IGM_SKETCH_DIM:-32}" \
            --set defense_setting.spectra.igm_sketch_seed="${SPECTRA_IGM_SKETCH_SEED:-271828}" \
            --set defense_setting.spectra.identifiability_weight="${SPECTRA_IDENTIFIABILITY_WEIGHT:-3.0}" \
            --set defense_setting.spectra.match_floor_weight="${SPECTRA_MATCH_FLOOR_WEIGHT:-2.0}" \
            --set defense_setting.spectra.gradient_cosine_weight="${SPECTRA_GRADIENT_COSINE_WEIGHT:-0.25}" \
            --set defense_setting.spectra.seed="$seed") ;;

        none) defense_args=(--set defense_method=null) ;;
        noise) defense_args=(--set defense_method=noise --set defense_setting.noise="$NOISE_STD") ;;
        clipping) defense_args=(--set defense_method=clipping --set defense_setting.clipping="$CLIP_BOUND") ;;
        sparsi) defense_args=(--set defense_method=compression --set defense_setting.compression="$SPARSI_PERCENT") ;;
        soteria) defense_args=(--set defense_method=representation --set defense_setting.representation=10) ;;
        censor) defense_args=(--set defense_method=orthogonal --set our_num_tries="$CENSOR_TRIALS") ;;

        *) echo "Unsupported defense: $method" >&2; exit 4 ;;
      esac

      echo "[RUN] seed=$seed attack=$attack defense=$method"
      "$PYTHON" run_rec.py --config default_config.yml \
        --set exp_name="imagenet_${attack}_${method}_seed${seed}" \
        --set output_dir="$RESULTS_DIR" --set data_path="$DATA_PATH" \
        --set dataset=IMAGENET_IO --set model=ResNet18ImageNet \
        "${attack_args[@]}" "${defense_args[@]}" "${model_args[@]}" \
        --set num_images=1 --set num_exp="$NUM_EXP" \
        --set target_id=0 --set "target_indices=$target_indices_yaml" --set target_stride=1000 \
        --set restarts="$RESTARTS" --set train_epochs=1 \
        --set accuracy_eval_batches="$ACC_BATCHES" --set post_attack_train_batches=0 \
        --set set_seed="$seed" --set attack_seed="$seed"
      date -u +'%Y-%m-%dT%H:%M:%SZ' > "$marker"
    done
  done
done

"$PYTHON" visualize_cifar10_results.py \
  --input-root "$RESULTS_DIR" --output-dir "$OUT/$RUN_ID/visualization" --expected-samples "$NUM_EXP"

echo "Results: $OUT/$RUN_ID"
