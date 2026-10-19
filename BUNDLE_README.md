# PTGP and comparison-method source bundle

This source-only bundle contains the PTGP implementation (`ptgp/`, `train.py`, `privacy_adapter.py`), the original GMM comparison evaluator (`run_rec.py`, `defense.py`, `spectra.py`, `gpu_math.py`, `inversefed/`, `dnnlib/`, `torch_utils/`), baseline and attack configuration, and selected experiment drivers under `experiments/`. The original comparison defenses include no defense, noise, clipping, compression/Sparsi, representation pruning/Soteria, orthogonal-gradient/CENSOR, and the GMM/Spectra variants. The evaluator provides the original IG, GI, GGL, GIAS, and GIFD attack paths.

The bundle intentionally omits datasets, outputs, model and generator weights, downloaded archives, editor settings, caches, and unrelated historical experiment variants. The selected experiment scripts preserve command-line settings and may contain paths specific to the original server; set dataset, checkpoint, and output paths for the target machine before running them. Generator-based attacks require their original external checkpoints.

For ImageNet-1K, the classifier-training directory is `train/<WNID>/...`. The evaluation/test directory in this codebase is `val/<WNID>/...`; PTGP also chooses its reference images from `val/`, excluding the exact attacked target indices. Both the PTGP loader and the bundled `IMAGENET_IO` evaluator require `train/` and `val/`, with matching class mappings. Neither uses `val/` as a training fallback. This is a transductive reference protocol and should be reported as such when comparing methods.

Smoke checks: `python -m unittest discover -s tests -v` and `python -m compileall -q ptgp run_rec.py defense.py inversefed`.
