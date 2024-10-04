#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="emma-7b-v1"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$VAST/eval/gqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa_loader \
        --model-path emma-7b \
        --question-file $VAST/eval/gqa/$SPLIT.jsonl \
        --image-folder $VAST/eval/gqa/data/images \
        --answers-file $VAST/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$VAST/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $VAST/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

$SCRATCH/aa10460/pytorch-example/python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
$SCRATCH/aa10460/pytorch-example/python eval.py --tier testdev_balanced
