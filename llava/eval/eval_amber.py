from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import torchvision.transforms as transforms

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image


class AmberDataset(Dataset):
    def __init__(self, image_path, query, annotation):
        
        self.image_path = image_path
        #self.query, self.annotation = list(), list()
        self.is_generative = True if 'generative' in query else False
        with open(query) as file:
            self.query = json.load(file)
          
        with open(annotation) as file:
            self.annotation = json.load(file)
        print(f'annotation loaded: {len(self.annotation)}')
        
        if self.is_generative:
            self.query = self._post_process_query()
            
        print(f'query loaded: {len(self.query)}') 
    
    def _post_process_query(self):
        new_query_list = list()
        for q in self.query:
                if 'truth' in self.annotation[q['id']]:
                    for idx, obj in enumerate(self.annotation[q['id']]['truth']):
                        new_query_list.append({'id': q['id'], 'type':'truth', 'type_id':idx, 'image': q['image']})
                if 'hallu' in self.annotation[q['id']]:
                    for idx, obj in enumerate(self.annotation[q['id']]['hallu']):
                        new_query_list.append({'id': q['id'], 'type':'hallu', 'type_id':idx , 'image': q['image']})
        return new_query_list
    
    
    def __getitem__(self, idx):
        data = self.query[idx]
        if self.is_generative:
            id, type_, type_id, image = data['id'], data['type'], data['type_id'], data['image']
            query = 'Is {obj} in this image?'.format(obj=self.annotation[id][type_][type_id])
            answer = 'yes' if type_ == 'truth' else 'no'
        else:
            id, image, query = data['id'], data['image'], data['query']
            # image = process_images([image], self.image_processor, self.model_config)[0]
            answer = self.annotation[id]['truth']
        image = Image.open(os.path.join(self.image_path, image)).convert('RGB')
        return id, query, image, answer
        
        
    def __len__(self):
        return len(self.query)

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, cache_dir='./')

    amber_dataset = AmberDataset(image_path=args.image_folder, query=args.question_file, annotation=args.answers_file) 

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
        
    count, all = 0, 0
    answers = dict()
    prev_id = ''
    for id_, question, image, answer in tqdm(amber_dataset):   
        print(id_)
        qs = cur_prompt = question
        prompt_qs = qs.split('\n')[0]
        prompt_input_ids = model.get_vision_tower().tokenizer(prompt_qs) 
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        qs = qs + '\n' + "Answer with yes or no"

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
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
        pred = (outputs.lower() == answer.lower())
        
        if id_ != prev_id:
            count += 1 if len(answers) > 0  and False not in answers[prev_id] else 0
            prev_id = id_
            answers[id_] = set()
            answers[id_].add(pred)
            all += 1
            print(f'count: {count}, all: {all}, acc:{round(count/all, 4)}, output: {outputs.lower()}, answer: {answer.lower()}')

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
