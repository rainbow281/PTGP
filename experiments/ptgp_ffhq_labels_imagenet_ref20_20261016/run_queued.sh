#!/bin/bash
set -euo pipefail
exec >> /mnt/sda/rainbow/yanch_gmm/privacy_ptgp_ffhq_labels_imagenet_ref20_gpu1_20261016/logs/matrix.log 2>&1
echo "WAITING for original 140-job matrix to finish"
while [ ! -f /mnt/sda/rainbow/yanch_gmm/privacy_single_energy_all_methods_doc_fullrerun_gpu1_20261015/COMPLETE ]; do
  if [ -f /mnt/sda/rainbow/yanch_gmm/privacy_single_energy_all_methods_doc_fullrerun_gpu1_20261015/ERROR ]; then echo "Original matrix failed; corrected run not started"; exit 1; fi
  sleep 30
done
echo "START corrected FFHQ and ImageNet experiment"
exec /home/yanch/GMM/experiments/ptgp_ffhq_labels_imagenet_ref20_20261016/run.sh
