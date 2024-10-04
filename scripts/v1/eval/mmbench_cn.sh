#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path emma-7b \
    --question-file $VAST/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file $VAST/eval/mmbench_cn/answers/$SPLIT/emma-7b-v1.jsonl \
    --lang cn \
    --attack none \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --all-rounds

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $VAST/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir $VAST/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir $VAST/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment emma-7b-v1
