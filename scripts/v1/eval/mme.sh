#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path emma-7b  \
    --question-file $VAST/eval/MME/llava_mme.jsonl \
    --image-folder $VAST/eval/MME/MME_Benchmark_release_version \
    --answers-file $VAST/eval/MME/answers/emma-7b-v1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd $VAST/eval/MME

python convert_answer_to_mme.py --experiment emma-7b-v1

cd eval_tool

python calculation.py --results_dir answers/emma-7b-v1
