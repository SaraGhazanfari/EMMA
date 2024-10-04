import argparse
import json
import math
import os
import ast
import pandas as pd
# import shortuuid
import torch
from tqdm import tqdm
from datasets import load_dataset
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

all_options = ['A', 'B', 'C', 'D']



def eval_model(args):
    ds = load_dataset("q-future/A-Bench-HF", cache_dir='/vast/sg7457')
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')



    count, all = 0, 0

    
    #{'id': 0, 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x288 at 0x7F0240055F90>, 
    # 'question': 'May I ask where the scene in the picture is located?', 'option0': 'Forest', 'option1': 'Riverside', 
    # 'option2': 'Desert', 'option3': 'N/A', 'category': 'part1 -> bag_of_words -> attribute', 'correct_choice': 'B'}
    for row in tqdm(ds['dev']):
        options = []
        for option_id in range(4):
            if row[f'option{option_id}'] != 'N/A':
                options.append(row[f'option{option_id}'])
                
        cur_option_char = all_options[:len(options)]

        question = row['question']
        image = row['image'].convert('RGB')
            
        for option_char, option in zip(all_options[:len(options)], options):
            question = question + '\n' + option_char + '. ' + option
        qs = cur_prompt = question
        
        prompt_qs = qs.split('\n')[0]
        prompt_input_ids = model.get_vision_tower().tokenizer(prompt_qs) 
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        image_tensor = process_images([image], image_processor, model.config)

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor.half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                prompt_input_ids=prompt_input_ids.repeat(image_tensor.shape[0],1))

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        count += (outputs == row['correct_choice'])
        all += 1
        print(f'count: {count}, all: {all}, acc: {round(count/all, 4)} outputs: {outputs}, answer: {row["correct_choice"]}')
        # rotate options
        options = options[1:] + options[:1]
        cur_option_char = cur_option_char[1:] + cur_option_char[:1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
