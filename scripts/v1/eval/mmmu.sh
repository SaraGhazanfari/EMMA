#!/bin/bash

SPLIT="mmbench_dev_20230712"

$SCRATCH/code/final/pytorch-example/python -m llava.eval.model_vqa_mmmu \
    --model-path emma-7b \
    --question-file $VAST/eval/mmmu \
    --answers-file $VAST/eval/mmmu/answers/$SPLIT/emma-7b-v1.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

