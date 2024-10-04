#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path emma-7b \
    --question-file $VAST/eval/vizwiz/llava_test.jsonl \
    --image-folder $VAST/eval/vizwiz/test \
    --answers-file $VAST/eval/vizwiz/answers/emma-7b-v1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $VAST/eval/vizwiz/llava_test.jsonl \
    --result-file $VAST/eval/vizwiz/answers/emma-7b-v1.jsonl \
    --result-upload-file $VAST/eval/vizwiz/answers_upload/emma-7b-v1.json
