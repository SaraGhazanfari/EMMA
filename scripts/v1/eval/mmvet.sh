#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path emma-7b \
    --question-file $VAST/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $VAST/eval/mm-vet/images \
    --answers-file $VAST/eval/mm-vet/answers/emma-7b-v1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $VAST/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $VAST/eval/mm-vet/answers/emma-7b-v1.jsonl \
    --dst $VAST/eval/mm-vet/results/emma-7b-v1.json

