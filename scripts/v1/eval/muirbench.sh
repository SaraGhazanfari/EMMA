$SCRATCH/code/final/pytorch-example/python -m llava.eval.eval_muirbench \
    --model-path emma-7b \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1