"""Run seven defenses under one CIFAR-10 federated training protocol."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["SPECTRA_EXPERIMENT_ROOT"])
HERE = Path(__file__).resolve().parent
PYTHON = "/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python"
METHODS = ["spectra", "censor", "none", "noise", "clipping", "sparsi", "soteria"]


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def main():
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    for position, method in enumerate(METHODS, 1):
        out = ROOT / "methods" / method
        out.mkdir(parents=True, exist_ok=True)
        if (out / "COMPLETE").exists():
            continue
        atomic_json(ROOT / "stage.json", dict(method=method, position=position,
                                             total=len(METHODS), phase="RUNNING", time=time.time()))
        command = [PYTHON, str(HERE / "train.py"), "--method", method,
                   "--rounds", "2000", "--seed", "1234", "--checkpoint-every", "10",
                   "--output", str(out)]
        print("START", method, flush=True)
        with (ROOT / "logs" / f"{method}.log").open("a") as log:
            result = subprocess.run(command, cwd="/home/yanch/GMM", stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            atomic_json(ROOT / "stage.json", dict(method=method, position=position,
                                                 phase="FAILED", exit_code=result.returncode))
            raise RuntimeError(f"{method} failed; inspect {ROOT / 'logs' / (method + '.log')}")
        print("COMPLETE", method, flush=True)
    (ROOT / "COMPLETE").write_text("Seven methods x 2000 rounds\n")
    atomic_json(ROOT / "stage.json", dict(phase="COMPLETE", time=time.time()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (ROOT / "ERROR").write_text(repr(error))
        raise
