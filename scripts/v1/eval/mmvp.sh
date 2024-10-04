$SCRATCH/aa10460/pytorch-example/python -m llava.eval.eval_mmvp \
    --model-path emma-7b \
    --question-file $VAST/eval/vision-bench/MMVP/Questions.csv \
    --image-folder "$VAST/eval/vision-bench/MMVP/MMVP Images" \
    --answers-file $VAST/eval/vision-bench/MMVP/answers/emma-7b-v1.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1
