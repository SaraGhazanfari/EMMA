#!/bin/bash

python scripts/v1/eval/save_config.py

CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/vizwiz.sh
echo "vizwiz is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/sqa.sh
echo "sqa is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/gqa.sh
echo "gqa is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/pope.sh
echo "pope is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/mmbench.sh
echo "mmbench is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/mmbench_cn.sh
echo "mmbench_cn is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/mmmu.sh
echo "mmmu is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/qbench.sh
echo "qbench is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/abench.sh
echo "abench is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/mmvp.sh
echo "MMVP is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/mathvista.sh
echo "mathvista is finished"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/vqav2.sh
echo "vqav2 is finished"



