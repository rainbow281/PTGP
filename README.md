# PTGP: Principal Task Gradient Projection

**Task-guided gradient inversion defense for federated learning**

PTGP protects shared client gradients by retaining the dominant task-relevant update direction rather than applying large, indiscriminate gradient perturbations. A class-balanced reference set supplies task-gradient probes; their principal direction is extracted and used to project client gradients before aggregation. The result is still a gradient with the same parameter structure, compatible with a standard FedAvg update interface.

> **Reproducibility note.** This repository is a source-code release, not a self-contained reproduction of all reported experiments. Datasets, pretrained classifier/generator checkpoints, experiment outputs and numerical tables are not bundled. Historical scripts may need path and dependency adjustments. Do not treat a successful smoke test as verification of the paper's numerical results.

## 1. Method overview

Let the current model be parameterized by $\theta$ and let $R_1,\ldots,R_m$ be class-balanced reference groups. Compute

$$q_j=\nabla_\theta L(\theta;R_j),\qquad Q=[q_1^\top;\ldots;q_m^\top].$$

Let $v_1$ be the leading eigenvector of $QQ^\top$. The normalized principal direction in parameter space is

$$u=\frac{Q^\top v_1}{\|Q^\top v_1\|_2},\qquad \widetilde g=(g^\top u)u.$$

Only **one direction** is retained. The released implementation uses no gradient quantization, random supplementary directions or projected-norm clipping. Reference image indices are fixed for a run, while the direction is recomputed at the current model state during federated training. For a fixed-model, single-round attack evaluation, it is fitted once.

**Implementation defaults:** 32 reference groups; seed `1234`; one retained direction. The default reference budget is 200 images per class for all four datasets. The implementation computes the probe Gram matrix on CPU and preserves model state while evaluating probes.

## 2. Repository map

| Path | Purpose |
| --- | --- |
| `ptgp/core.py` | Balanced reference groups, task-gradient probes, principal direction and projection |
| `ptgp/data.py` | Dataset loaders and explicit training/reference/evaluation splits |
| `ptgp/model.py` | CIFAR-style ResNet-18 model constructor |
| `train.py` | Minimal FedAvg experiment: PTGP versus no defense |
| `privacy_adapter.py` | Injects PTGP into the included legacy attack evaluator |
| `run_rec.py`, `defense.py` | GMM/CENSOR attack and defense evaluation backend with a PTGP entry point |
| `inversefed/`, `dnnlib/`, `torch_utils/` | Supporting attack, generator and utility implementations |
| `default_config.yml` | Legacy attack evaluator configuration |
| `tools/` | Dataset preparation and validation helpers |
| `tests/test_ptgp.py` | Projection identity and ImageNet split-safety tests |
| `tests/` | Unit tests for the projection and split protocol |

**Important:** `privacy_adapter.py` registers `defense.ptgp_gradient` and selects `defense_method="ptgp"` for attack evaluation. Use `train.py` and `privacy_adapter.py` as the PTGP entry points.

## 3. Environment and quick verification

Use Python 3.9+ and a PyTorch/torchvision pair compatible with your CUDA driver. The minimal requirements are in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

These dependencies cover the core method; the five-attack evaluator and generator-based attacks can require additional packages and external checkpoints. The tests check mathematical and data-split invariants, not paper-level accuracy or privacy results.

## 4. Dataset protocol

**Required protocol for the paper:** use **200 class-balanced reference images per class** for every dataset. For imagenet, both classifier training and PTGP reference-gradient construction must use the official **`train/` split, not `val/`**. Keep evaluation images distinct from the reference images when reporting privacy results.

| Dataset | Classifier training | PTGP reference source | Evaluation source | Reference images per class |
| --- | --- | --- | --- | ---: |
| CIFAR-10 | official `train` | official `train` | official `test` | 200 |
| CIFAR-100 | official `train` | official `train` | official `test` | 200 |
| FFHQ (10 age classes) | labeled `train/` | labeled `train/` | labeled `eval/` | 200 |
| imagenet | official `train/` | official `train/` | official `val/` for evaluation only | 200 |

Thus the reference budgets are 2,000 images for CIFAR-10, 20,000 for CIFAR-100, 2,000 for labeled FFHQ, and **200,000 for imagenet**. Reference images are selected class-by-class without replacement, with a fixed random seed. The same reference set is used to compute 32 task-gradient probes; the final projection retains only the leading task direction. The number of reference images is distinct from the number of attack targets.

**ImageNet data organization:**

```text
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1,000 WNID class directories)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (evaluation only)
```

The validation split must **not** be used to construct the PTGP task direction or to train the classifier. In a single-round reconstruction evaluation, use held-out target images from the evaluation split, with no overlap with the reference set. The ImageNet class mappings must be consistent between splits.

**Implementation:** `ptgp/data.py` returns `(train, train, val)` for ImageNet; `train.py` and `privacy_adapter.py` default to 200 reference images per class on all four datasets. ImageNet `val/` is reserved for evaluation. Full end-to-end attack reproduction still requires external datasets, checkpoints and matching evaluator settings.

## 5. Federated-training example

`train.py` is a compact FedAvg driver, not an exact reproduction of every historical training configuration. Defaults: randomly initialized CIFAR-style ResNet-18, 100 clients, 10 sampled clients per round, one local epoch, batch size 64, SGD learning rate 0.1, Dirichlet concentration 0.5 and 2,000 rounds. It writes per-round loss/accuracy to `round_metrics.csv`.

Start with a short smoke test (datasets must already exist):

```bash
python train.py --dataset cifar10 --data-root /path/to/cifar-data \
  --method ptgp --rounds 2 --output ./outputs/ptgp_smoke

python train.py --dataset cifar10 --data-root /path/to/cifar-data \
  --method none --rounds 2 --output ./outputs/baseline_smoke
```

For a longer run, change `--rounds 2` to the desired count, e.g. `--rounds 2000`. `--dataset` also accepts `cifar100`, `ffhq`, and `imagenet`, subject to the split requirements above. ImageNet and the full attack matrix can require substantial GPU time, memory and storage.

## 6. Single-round gradient-inversion evaluation

The legacy evaluation backend includes attack implementations intended for IG/Geiping, GI, GGL, GIAS and GIFD, with baseline defenses including no defense, noise, clipping, sparsification, Soteria and CENSOR. Attack settings are controlled by the legacy `run_rec.py` configuration; generator-based methods need their original compatible generator weights.

Example invocation of the PTGP adapter for ImageNet (replace all paths, and confirm the evaluator accepts the chosen attack flags):

```bash
python privacy_adapter.py \
  --gmm-root . \
  --dataset imagenet \
  --data-root /path/to/imagenet \
  --target-indices 0 1000 2000 3000 4000 5000 6000 7000 8000 9000 \
  --config ./default_config.yml \
  --batch_size 1 \
  --set dataset='"IMAGENET_IO"' \
  --set trained_model=false \
  --set epochs=1 \
  --set geiping=true
```

The `--dataset` flag on the adapter is lowercase; the legacy evaluator uses identifiers such as `CIFAR10`, `CIFAR100`, `FFHQ64` and `IMAGENET_IO`. For ImageNet, verify that the legacy evaluator actually draws attack targets from `val/`, while reference probes use `train/`, and that target/reference samples do not overlap. The adapter does not automatically make every historical experiment script PTGP-aware.

**Evaluation checklist:** hold fixed model initialization, attacked image IDs, batch size, preprocessing, attack iterations/restarts, generator checkpoints and label mapping across defenses. Evaluate the strongest relevant attack configurations rather than comparing unlike budgets. Report MSE and LPIPS (**higher is better for privacy**), PSNR and SSIM (**lower is better for privacy**), and classifier ACC (**higher is better for utility**). Specify LPIPS backbone(s), whether VGG/AlexNet values were averaged, aggregation level and standard deviation definition.

## 7. Scope, limitations and citation

- Projection preserves a task-aligned update component but does **not** establish formal differential privacy or a universal guarantee against reconstruction.
- Reference distribution, model state and probe configuration can influence both privacy and accuracy; the one-direction design may discard useful task updates.
- This release does not contain datasets, weights, experimental outputs, or verified numerical reproductions of the paper tables.
- The source bundle retains necessary GMM attack backend utilities; distinguish them from the final PTGP implementation when reusing or citing code.

**Paper and citation:** add the final paper title, author list, venue/year and BibTeX entry after publication or as permitted by the review policy. Do not invent bibliographic metadata. For anonymous review, ensure repository ownership, commit history, embedded paths and external links comply with the venue's anonymity requirements.

## Repository cleanup

Historical dated experiment runs and obsolete comparison launchers were removed. Core PTGP, the attack backend, generator support, dataset utilities and tests remain. Some legacy backend identifiers and external data paths must be configured to reproduce paper-scale results.

## Release identity

The proposed defense method is **PTGP**. The evaluation interface and documentation use the same method identifier.
