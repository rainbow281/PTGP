#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-/mnt/sda/rainbow/yanch_gmm/data/ffhq_eval}"
OUT="${OUT:-/home/yanch/GMM/comparison_results/ffhq_five_attack_full}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/sda/rainbow/yanch_gmm/cache/ffhq}"
STYLEGAN2_CHECKPOINT="${STYLEGAN2_CHECKPOINT:-/mnt/sda/rainbow/yanch_gmm/models/ffhq_stylegan2/stylegan2-ffhq-config-f.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/mnt/sda/rainbow/yanch_gmm/models/ffhq_resnet18.pth}"
AUTO_TRAIN_FFHQ_CLASSIFIER="${AUTO_TRAIN_FFHQ_CLASSIFIER:-true}"
FFHQ_TRAIN_CACHE="${FFHQ_TRAIN_CACHE:-$(dirname "$DATA_PATH")/ffhq_train64}"
FFHQ_CLASSIFIER_EPOCHS="${FFHQ_CLASSIFIER_EPOCHS:-50}"
FFHQ_CLASSIFIER_BATCH_SIZE="${FFHQ_CLASSIFIER_BATCH_SIZE:-128}"
FFHQ_CLASSIFIER_WORKERS="${FFHQ_CLASSIFIER_WORKERS:-4}"
GAUSSIAN_FIT="${GAUSSIAN_FIT:-$PWD/gaussian_fit.pt}"
ATTACKS="${ATTACKS:-geiping yin ggl gias gifd}"
METHODS="${METHODS:-none noise clipping sparsi soteria censor spectra}"
FULL_MATRIX="${FULL_MATRIX:-true}"
if [[ "$FULL_MATRIX" == true ]]; then
  [[ "$ATTACKS" == "geiping yin ggl gias gifd" ]] || { echo "Experiment notice: " >&2; exit 2; }
  [[ "$METHODS" == "none noise clipping sparsi soteria censor spectra" ]] || { echo "Experiment notice: " >&2; exit 2; }
fi
SEEDS="${SEEDS:-1234}"
TARGET_INDICES="${TARGET_INDICES:-0 1 2 3 4 5 6 7 8 9}"
NUM_EXP="${NUM_EXP:-10}"
RESTARTS="${RESTARTS:-1}"
PIXEL_ITERS="${PIXEL_ITERS:-1000}"
GAN_ITERS="${GAN_ITERS:-300}"
GGL_ITERS="${GGL_ITERS:-300}"
GGL_CMA_BUDGET="${GGL_CMA_BUDGET:-150}"
GGL_NUM_SAMPLES="${GGL_NUM_SAMPLES:-10}"
GIAS_LATENT_ITERS="${GIAS_LATENT_ITERS:-300}"
GIAS_PARAM_ITERS="${GIAS_PARAM_ITERS:-300}"
GIFD_STAGE_ITERS="${GIFD_STAGE_ITERS:-50}"
NOISE_STD="${NOISE_STD:-0.01}"
CLIP_BOUND="${CLIP_BOUND:-4}"
SPARSI_PERCENT="${SPARSI_PERCENT:-20}"
CENSOR_TRIALS="${CENSOR_TRIALS:-20}"
GMM_CANDIDATES="${GMM_CANDIDATES:-5}"
ACC_BATCHES="${ACC_BATCHES:-1}"

[[ "$DATA_PATH" = /* && "$OUT" = /* ]] || { echo "Experiment notice: " >&2; exit 2; }
[[ -f "$STYLEGAN2_CHECKPOINT" ]] || { echo "Experiment notice: $STYLEGAN2_CHECKPOINT" >&2; exit 3; }
if [[ ! -f "$MODEL_CHECKPOINT" && "$AUTO_TRAIN_FFHQ_CLASSIFIER" == true ]]; then
  echo "Experiment notice: "
  "$PYTHON" train_ffhq_classifier.py \
    --eval-manifest "$DATA_PATH/ffhq_eval_manifest.json" \
    --cache-root "$FFHQ_TRAIN_CACHE" --output "$MODEL_CHECKPOINT" \
    --epochs "$FFHQ_CLASSIFIER_EPOCHS" --batch-size "$FFHQ_CLASSIFIER_BATCH_SIZE" \
    --workers "$FFHQ_CLASSIFIER_WORKERS" --seed 1234 --image-size 64
fi
[[ -f "$MODEL_CHECKPOINT" ]] || { echo "Experiment notice: $MODEL_CHECKPOINT" >&2; exit 3; }
[[ -f "$GAUSSIAN_FIT" ]] || { echo "Experiment notice: $GAUSSIAN_FIT" >&2; exit 3; }

RESULTS="$OUT/results"; DONE="$OUT/completed"
mkdir -p "$RESULTS" "$DONE" "$CACHE_ROOT"/torch/hub/checkpoints "$CACHE_ROOT"/{matplotlib,xdg,pycache,tmp}
export TORCH_HOME="$CACHE_ROOT/torch" MPLCONFIGDIR="$CACHE_ROOT/matplotlib" XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache" TMPDIR="$CACHE_ROOT/tmp" MPLBACKEND=Agg GIA_NUM_WORKERS="${GIA_NUM_WORKERS:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-1234}" CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
for weight in vgg16-397923af.pth alexnet-owt-7be5be79.pth; do
  [[ -f "$TORCH_HOME/hub/checkpoints/$weight" ]] || cp "${PRETRAINED_CACHE_SOURCE:-$HOME/.cache/torch/hub/checkpoints}/$weight" "$TORCH_HOME/hub/checkpoints/$weight"
done
read -r -a ids <<< "$TARGET_INDICES"
[[ "${#ids[@]}" -eq "$NUM_EXP" ]] || { echo "Experiment notice: " >&2; exit 4; }
ids_yaml="[$(IFS=,; echo "${ids[*]}")]"

for seed in $SEEDS; do
  for attack in $ATTACKS; do
    case "$attack" in
      geiping) attack_args=(--set optim=GAN_free --set geiping=true --set yin=false --set ggl=false --set gias=false --set gifd=false --set generative_model= --set max_iterations="$PIXEL_ITERS") ;;
      yin) attack_args=(--set optim=GAN_free --set geiping=false --set yin=true --set ggl=false --set gias=false --set gifd=false --set generative_model= --set max_iterations="$PIXEL_ITERS") ;;
      ggl) attack_args=(--set optim=GAN_based --set generative_model=stylegan2_io --set gen_dataset=FFHQ64 --set ckpt="$STYLEGAN2_CHECKPOINT" --set geiping=false --set yin=false --set ggl=true --set gias=false --set gifd=false --set start_layer=0 --set end_layer=8 --set max_iterations="$GGL_ITERS" --set cma_budget="$GGL_CMA_BUDGET" --set num_sample="$GGL_NUM_SAMPLES") ;;
      gias) attack_args=(--set optim=GAN_based --set generative_model=stylegan2_io --set gen_dataset=FFHQ64 --set ckpt="$STYLEGAN2_CHECKPOINT" --set geiping=false --set yin=false --set ggl=false --set gias=true --set gifd=false --set start_layer=0 --set end_layer=8 --set max_iterations="$GIAS_LATENT_ITERS" --set gias_iterations="$GIAS_PARAM_ITERS" --set gias_lr=0.00001) ;;
      gifd)
        attack_args=(--set optim=GAN_based --set generative_model=stylegan2_io --set gen_dataset=FFHQ64 --set ckpt="$STYLEGAN2_CHECKPOINT" --set geiping=false --set yin=false --set ggl=false --set gias=false --set gifd=true --set max_iterations="$GAN_ITERS" --set start_layer=0 --set end_layer=8 --set "steps=[$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS,$GIFD_STAGE_ITERS]" --set 'lr_io=[0.1,0.1,0.08,0.06,0.04,0.03,0.02,0.01,0.005]' --set project=false --set do_project_gen_out=false --set do_project_noises=false --set do_project_latent=false --set 'max_radius_gen_out=[-1,-1,-1,-1,-1,-1,-1,-1,-1]' --set 'max_radius_noises=[-1,-1,-1,-1,-1,-1,-1,-1,-1]' --set 'max_radius_latent=[-1,-1,-1,-1,-1,-1,-1,-1,-1]') ;;
      *) echo "Experiment notice: $attack" >&2; exit 5 ;;
    esac
    for method in $METHODS; do
      marker="$DONE/v2_seed${seed}_${attack}_${method}.done"
      [[ -s "$marker" ]] && grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' "$marker" && { echo "[SKIP] $marker"; continue; }
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

        *) echo "Experiment notice: $method" >&2; exit 6 ;;
      esac
      echo "[RUN] FFHQ seed=$seed attack=$attack defense=$method"
      "$PYTHON" run_rec.py --config default_config.yml --set exp_name="ffhq_${attack}_${method}_seed${seed}" --set output_dir="$RESULTS" --set data_path="$DATA_PATH" --set dataset=FFHQ64 --set model=ResNet18 "${attack_args[@]}" "${defense_args[@]}" --set trained_model=true --set model_checkpoint_path="$MODEL_CHECKPOINT" --set num_images=1 --set num_exp="$NUM_EXP" --set target_indices="$ids_yaml" --set target_id=0 --set restarts="$RESTARTS" --set train_epochs=1 --set accuracy_eval_batches="$ACC_BATCHES" --set post_attack_train_batches=0 --set set_seed="$seed" --set attack_seed="$seed"
      date -u +'%Y-%m-%dT%H:%M:%SZ' > "$marker"
    done
  done
done
"$PYTHON" visualize_cifar10_results.py --input-root "$RESULTS" --output-dir "$OUT/visualization" --expected-samples "$NUM_EXP"
echo "FFHQ results: $OUT"
