#!/bin/bash
set -euo pipefail
BASE=/mnt/sda/rainbow/yanch_gmm/ptgp_ablation_cifar10_gpu1_20261017
ROOT=/mnt/sda/rainbow/yanch_gmm/ptgp_min_energy_cifar10_gpu1_20261017
echo "$(date -Is) queued: waiting for original privacy screen to finish" | tee -a "$ROOT/logs/screen.log"
while screen -ls | grep -q 'ptgp_ablation_cifar10_gpu1'; do sleep 30; done
echo "$(date -Is) starting minimum-energy comparison" | tee -a "$ROOT/logs/screen.log"
exec bash /home/yanch/GMM/experiments/ptgp_min_energy_cifar10_20261017/run.sh >> "$ROOT/logs/screen.log" 2>&1
