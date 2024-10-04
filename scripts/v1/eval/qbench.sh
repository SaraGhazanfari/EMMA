python -m llava.eval.model_vqa_qbench \
    --model-path emma-7b \
    --question-file $VAST/eval/qbench/llvisionqa_dev.json \
    --image-folder $VAST/eval/qbench/images \
    --answers-file $VAST/eval/qbench/answers/emma-7b-v1.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1
# $SCRATCH/aa10460/pytorch-example/python -m llava.eval.eval_textvqa \
#     --annotation-file $VAST/eval/qbench/llvisionqa_dev.json \
#     --result-file $VAST/eval/textvqa/answers/emma-7b-v1.jsonl