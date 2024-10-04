#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path emma-7b \
    --question-file $VAST/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $VAST/eval/scienceqa/images/test \
    --answers-file $VAST/eval/scienceqa/answers/emma-7b-v1.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir $VAST/eval/scienceqa \
    --result-file $VAST/eval/scienceqa/answers/emma-7b-v1.jsonl \
    --output-file $VAST/eval/scienceqa/answers/emma-7b-v1_output.jsonl \
    --output-result $VAST/eval/scienceqa/answers/emma-7b-v1_result.json
