# PTGP: Principal Task Gradient Projection


PTGP is a compact research implementation of **single-direction task-gradient projection** for gradient-inversion defense. Given class-balanced reference images, it forms several task-gradient probes, extracts their leading principal direction, and projects each client gradient onto that direction. The transmitted object remains a parameter gradient. The implementation uses continuous coefficients; it does **not** quantize them, add random directions, or clip the projected norm.

This bundle contains the PTGP method, explicit dataset splits, a small federated-training driver, and the GMM/CENSOR comparison evaluator with its baseline defenses and IG, GI, GGL, GIAS, and GIFD attack code. It contains **no images, downloaded datasets, generator weights, model checkpoints, attack outputs, or numerical results**.

## Method

For model parameters \(\theta\), reference batches \(R_1,\ldots,R_m\), and cross-entropy loss \(L\), PTGP computes task probes

\[
q_j = \nabla_\theta L(\theta;R_j),\qquad
Q = [q_1^\top;\ldots;q_m^\top].
\]

It finds the top eigenvector \(v_1\) of \(QQ^\top\), then obtains the unit parameter-space direction \(u = Q^\top v_1 / \|Q^\top v_1\|_2\). For a client gradient \(g\), the defended gradient is

\[
\widetilde g = \langle g,u\rangle u.
\]

The reference image indices stay fixed during training; the gradient direction is recomputed at the current model state. For a one-round reconstruction experiment with a fixed random model, the direction is computed once. Probe forwards use training mode for CIFAR-10 and FFHQ, evaluation mode for CIFAR-100 and ImageNet, matching the remote PTGP variants. The implementation preserves BatchNorm buffers and restores the model's original train/eval flags after evaluating probes. It checks the projection identity and retained energy in the included unit test.

The main defaults follow the remote PTGP experiment: **32 class-mixed probe groups, one retained direction, seed 1234**. Reference counts are 200 per class for CIFAR-10, 100 per class for CIFAR-100, 200 per age class for the labeled FFHQ variant, and 20 per class for ImageNet-1K. These values are exposed as arguments, not silently inferred from available images.

## Data protocol — especially ImageNet

| Dataset | Model-training images | PTGP reference images | Evaluation images |
| --- | --- | --- | --- |
| CIFAR-10 | official `train` | class-balanced subset of `train` | official `test` |
| CIFAR-100 | official `train` | class-balanced subset of `train` | official `test` |
| FFHQ age-label variant | labeled `train/` | class-balanced subset of `train/` | labeled `eval/` |
| ImageNet-1K | **official `train/`** | **official `val/`**, used as the test/reference split | official `val/` |

ImageNet-1K's public validation directory is the **evaluation/test directory in this codebase**. The official test images are not used here because their labels are unavailable to this evaluation pipeline. No image from `val/` is used to train the classifier. The GMM evaluator's `IMAGENET_IO` loader and the PTGP loader both require `train/` and `val/`; neither substitutes `val/` for a missing training directory.

For ImageNet privacy evaluation, pass the attacked validation-image indices through `--target-indices`. PTGP excludes those exact images when selecting its validation reference set. Reference images still come from the same official split, so this is a **transductive reference protocol** and should be disclosed in any paper or comparison. It is not the same as an independent auxiliary training set.

ImageNet runs require both `train/` and `val/` directories with the same 1,000 class-directory mapping. Extract the training archive into `train/<WNID>/...`; an unextracted `ILSVRC2012_img_train.tar` is insufficient. The loader raises `FileNotFoundError` if either split is missing.

Expected ImageNet layout:

```text
imagenet/
  train/
    n01440764/...
    ... 1000 WNID directories ...
  val/
    n01440764/...
    ... 1000 WNID directories ...
```

The small FFHQ experiment uses **10 age-label classes**, not ordinary unlabeled FFHQ. Both `train/` and `eval/` must use folders `0` through `9` with identical meaning. The loader rejects a mismatched class mapping. Do not infer age labels from `ImageFolder` order unless the directories already encode the correct labels.

## Setup

Use Python 3.9 or newer and an appropriate PyTorch/CUDA build for the machine. Install the small method package dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The comparison evaluator is included in this bundle. Its attack-specific Python dependencies must be installed separately; GGL, GIAS, and GIFD also need their appropriate generator checkpoints. Point `privacy_adapter.py` to this bundle's root. Pretrained weights are not included.

## Federated-training example

The included `train.py` is a transparent FedAvg driver for PTGP versus an undefended control. Its defaults are a randomly initialized CIFAR-style ResNet-18, 100 clients, 10 sampled clients per round, one local epoch, batch size 64, SGD at learning rate 0.1, and a seeded class-wise Dirichlet partition with concentration 0.5. Each round records training and evaluation loss/accuracy in `round_metrics.csv`. This driver illustrates the method and is **not** a claim that every historical server run used this exact protocol.

```bash
python train.py \
  --dataset cifar10 --data-root /path/to/cifar-data \
  --method ptgp --rounds 2000 --output /path/to/output/ptgp

python train.py \
  --dataset cifar10 --data-root /path/to/cifar-data \
  --method none --rounds 2000 --output /path/to/output/no-defense
```

For ImageNet, set `--dataset imagenet --data-root /path/to/imagenet`. The command **requires** both `train/` and `val/`. Large-scale training requires substantial compute and storage; perform a short smoke run first.

## Privacy-evaluation adapter

`privacy_adapter.py` injects the PTGP gradient transform into a local copy of the original GMM/CENSOR evaluator, then executes its existing `run_rec.py`. All attack settings (`--config` and repeated `--set` arguments) are handled by that evaluator. The adapter enforces the selected data root and the PTGP defense method. Example for a one-round, batch-size-one attack run:

```bash
python privacy_adapter.py \
  --gmm-root . \
  --dataset imagenet \
  --data-root /path/to/imagenet \
  --target-indices 0 1000 2000 3000 4000 5000 6000 7000 8000 9000 \
  --config /path/to/GMM/default_config.yml \
  --batch_size 1 \
  --set dataset='"IMAGENET_IO"' \
  --set trained_model=false \
  --set epochs=1 \
  --set geiping=true
```

The attack evaluator controls target selection, attack iterations, generator choice, restarts, and metric export. Keep these settings identical across defenses. For CIFAR-10/100, use the evaluator's exact dataset identifier (`CIFAR10`/`CIFAR100`) while the adapter's `--dataset` uses lowercase. For FFHQ, use `FFHQ64` in the evaluator. When evaluating on ImageNet `val/`, the supplied `--target-indices` must exactly match the evaluator's target indices. The adapter makes no privacy guarantee and does not substitute one attack's result for another.

## Reported metrics and comparison rules

For each target image, record MSE and LPIPS (larger means poorer reconstruction), plus PSNR and SSIM (smaller means poorer reconstruction). If both VGG and AlexNet LPIPS implementations are present, report their arithmetic mean and say so. Report mean and sample standard deviation across the same target indices for every defense. Use a single classifier initialization and identical attack restarts, budgets, generator checkpoints, image preprocessing, and label mapping across compared methods.

Compute improvement percentages only from runs that use the same dataset splits, model initialization, attack settings, and label mapping.

## Repository contents

```text
ptgp/core.py        class-balanced groups, task-gradient probes, leading direction, projection
ptgp/data.py        explicit train/reference/evaluation splits and strict ImageNet checks
ptgp/model.py       random CIFAR-style ResNet-18 constructor
train.py            compact FedAvg training and per-round loss/accuracy output
privacy_adapter.py  hook into an external GMM/CENSOR five-attack evaluator
tests/              projection and ImageNet split-safety checks
requirements.txt    method-side dependencies
```

No author-identifying metadata, absolute personal filesystem paths, SSH credentials, datasets, model weights, or result files are included in the source tree. GitHub account ownership and repository creation metadata remain visible; a blind-review link requires an anonymization service or a dedicated review account.
