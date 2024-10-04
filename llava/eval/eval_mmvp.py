import argparse
import json
import math
import os
from PIL import Image
import pandas as pd
import shortuuid
import torch
from tqdm import tqdm
from llava.eval.visualize import visualize
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

    
def eval_model(args):
    # Model
    questions = pd.read_csv(os.path.expanduser(args.question_file))  # Assuming the fields are separated by tabs

    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    
    question_tokenizer = model.get_vision_tower().tokenizer
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    count, all = 0, 0
    result = [0,0]
    constant = [18, 19, 28, 29, 32, 33, 52, 53, 82, 83, 86, 87, 96, 97, 
                120, 121, 136, 137, 144, 145, 148, 149, 152, 153, 156, 
                157, 158, 159, 162, 163, 188, 189, 190, 191, 222, 223, 
                238, 239, 242, 243, 250, 251, 254, 255, 272, 273, 274, 
                275, 290, 291, 294, 295, 296, 297]

    for index, row in tqdm(questions.iterrows()):
        # Construct the 'prompts' string
        question = row['Question']
        print(question)
        options = row['Options'].split('(b)')
        options[0] = options[0].replace('(a)', '').strip()
        options[1] = options[1].strip()
        
        #get_options(row, all_options)
        cur_option_char = all_options[:len(options)]
        gt = 'A' if row['Correct Answer'] == '(a)' else 'B'
        
        for option_char, option in zip(all_options[:len(options)], options):
            question = question + '\n' + option_char + '. ' + option
        qs = cur_prompt = question
        
        prompt_qs = qs.split('\n')[0]
        prompt_input_ids = question_tokenizer(prompt_qs)

        image = Image.open(os.path.join(args.image_folder, f"{index+1}.jpg")).convert('RGB')
        
        image_tensor = process_images([image], image_processor, model.config)
        # previous_tokens = model.get_vision_tower().vision_tower(image_tensor.half().cuda(), 
        #                                                         output_hidden_states=True)
        # previous_tokens = torch.stack(previous_tokens.attentions).squeeze(1).detach().clone()
        # previous_attn = previous_tokens[-1, -1, -1, 1:].reshape(1, 24, 24) #torch.norm(previous_tokens[0, -1, 1:, 1:].reshape(576, 24, 24), p=2, dim=0)
        # current_tokens = previous_tokens[-1, -1, 1:, 1:].reshape(-1, 576)
        # weight = model.get_vision_tower().image_text_infusion.weight[:, :576]
        
        # current_tokens = torch.mul(current_tokens, weight.T).reshape(576, 24, 24)#+ model.get_vision_tower().image_text_infusion.bias
        # current_attn = current_tokens[-1].detach().clone()
        
        # print(current_attn.shape, previous_attn.shape)
        # if index in constant:
        #     visualize(tokens=current_attn.float(), img=image_tensor[0], save_path=f'all_pdf/{index}_emma_hm.pdf')
        #     visualize(tokens=previous_attn.float(), img=image_tensor[0], save_path=f'all_pdf/{index}_llava_hm.pdf')
        
        # #image.save(f"{index}.png")

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

        image_tensor = process_images([image], image_processor, model.config)[0]
        
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                prompt_input_ids=prompt_input_ids)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f'gt: {gt}:{row["Correct Answer"]} outputs: {outputs}')
        result[index%2] += (gt in outputs)
        if index % 2 == 1:
            count += (sum(result) // 2)
            all += 1
            result = [0,0]
            print(f'count:{count}, all: {all}, acc: {count/all}')



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