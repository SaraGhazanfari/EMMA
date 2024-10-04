#!/bin/bash
python -m llava.eval.eval_amber \
    --model-path emma-7b \
    --image-folder $VAST/eval/amber/image \
    --question-file $VAST/eval/amber/data/query/query_generative.json \
    --answers-file $VAST/eval/amber/data/annotations.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_amber \
    --model-path emma-7b \
    --image-folder $VAST/eval/amber/image \
    --question-file $VAST/eval/amber/data/query/query_discriminative-attribute.json \
    --answers-file $VAST/eval/amber/data/annotations.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

#query_discriminative-existence.json  query_discriminative.json  query_discriminative-relation.json
python -m llava.eval.eval_amber \
    --model-path emma-7b \
    --image-folder $VAST/eval/amber/image \
    --question-file $VAST/eval/amber/data/query/query_discriminative-existence.json \
    --answers-file $VAST/eval/amber/data/annotations.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_amber \
    --model-path emma-7b \
    --image-folder $VAST/eval/amber/image \
    --question-file $VAST/eval/amber/data/query/query_discriminative-relation.json \
    --answers-file $VAST/eval/amber/data/annotations.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1
