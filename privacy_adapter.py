"""Run the original five-attack GMM evaluator with the PTGP defense hook.

The historical attack implementations and generator weights are deliberately
external dependencies; no dataset, checkpoint, or result is bundled here.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

from ptgp import PTGP, load_dataset_splits, make_balanced_groups


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gmm-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "ffhq"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-indices", type=int, nargs="*", default=[])
    parser.add_argument("--reference-per-class", type=int)
    parser.add_argument("--probe-groups", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    arguments, attack_args = parser.parse_known_args()
    gmm_root = arguments.gmm_root.expanduser().resolve()
    if not (gmm_root / "run_rec.py").is_file():
        raise FileNotFoundError(f"Missing attack backend: {gmm_root / 'run_rec.py'}")
    # Strict split validation happens before any attack begins. In particular,
    # the ImageNet evaluator cannot silently train on val/ if train/ is absent.
    _, reference, _ = load_dataset_splits(arguments.dataset, str(arguments.data_root))
    per_class = arguments.reference_per_class or 200
    groups = make_balanced_groups(reference, per_class=per_class,
                                  groups=arguments.probe_groups,
                                  seed=arguments.seed,
                                  exclude_indices=())
    train_mode_probe = arguments.dataset in {"cifar10", "ffhq"}
    projector = PTGP(reference, groups,
                     microbatch_size=max(map(len, groups)) if train_mode_probe else 32,
                     probe_mode="train" if train_mode_probe else "eval")
    sys.path.insert(0, str(gmm_root))
    import defense  # pylint: disable=import-outside-toplevel

    def ptgp_gradient(gradients, model, _images, _labels, **_kwargs):
        if projector.basis is None:
            projector.fit(model)
        return projector.project(gradients)

    # PTGP evaluation entry point.
    defense.ptgp_gradient = ptgp_gradient
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    old_argv = sys.argv
    try:
        sys.argv = [str(gmm_root / "run_rec.py"), *attack_args,
                    "--set", "data_path=" + json.dumps(str(arguments.data_root.resolve())),
                    "--set", "defense_method=\"ptgp\""]
        runpy.run_path(str(gmm_root / "run_rec.py"), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
