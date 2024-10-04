#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path emma-7b \
    --question-file $VAST/eval/pope/llava_pope_test.jsonl \
    --image-folder $VAST/eval/pope/val2014 \
    --answers-file $VAST/eval/pope/answers/emma-7b-v1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir $VAST/eval/pope/coco \
    --question-file $VAST/eval/pope/llava_pope_test.jsonl \
    --result-file $VAST/eval/pope/answers/emma-7b-v1.jsonl
