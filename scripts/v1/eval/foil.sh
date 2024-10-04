python -m llava.eval.eval_foil \
    --model-path emma-7b \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1