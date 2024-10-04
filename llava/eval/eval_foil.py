import random
import argparse
from os.path import join
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from PIL import Image
import torch, os, pickle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path


DEFAULT_DIRS = {
    'images': '/coco/{split}2014/',  
    'val': 'foil/foilv1.0_test_2017.json',
    'train': 'foil/foilv1.0_train_2017.json',
    'hard-samples': 'foil/foil_hard_samples'
}

class Foil(Dataset):

    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None, instruct=None):
        self.split = split if split == 'train' else 'val'
        self.data_dir = data_dir
        self.annotation = self._read_annotation()
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.instruct = instruct
        with open(join(self.data_dir, DEFAULT_DIRS['hard-samples']), 'rb') as file:
            self.hard_samples = list(pickle.load(file))
    
    def _read_annotation(self):
        with open(join(self.data_dir, DEFAULT_DIRS[self.split])) as file:
            annotations = json.load(file)['annotations']
            
        self.annotations = dict()
        id_set = set()
        for ann in annotations:   
            id = ann['id']
            id_set.add(id)
            ann['image_id'] = f"COCO_{self.split}2014_{ann['image_id']:012d}.jpg" 
            self.annotations[id] = self.annotations.get(id, list())
            self.annotations[id].append(ann)
        self.id_list = list(id_set)   
                     
    def __getitem__(self, idx):
        samples = self.annotations[self.id_list[self.hard_samples[idx]]] #self.hard_samples[]
        caps = [sam['caption'] for sam in samples]
        label = 0 if not samples[0]['foil'] else 1
        img = Image.open(join(DEFAULT_DIRS['images'].format(split=self.split), 
                              samples[0]['image_id'])).convert('RGB') 
        
        label = random.randint(0, 1)  
        if label == 1:
            caps = caps[::-1]
            
        if self.instruct:
            caps = [self.instruct.format(cap1=caps[0], cap2=caps[1])]
            
        if self.text_preprocess is not None:
            caps = [self.text_preprocess(cap) for cap in caps]
            
        if self.img_preprocess is not None:
            img = self.img_preprocess(img)

        return img, *caps, label

    def __len__(self):
        return len(self.hard_samples)
    
def eval_model(args):
    # Model
    foil_dataset = Foil(data_dir='/vast/sg7457/uni_data', split='val')

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
                                                                           args.model_base, 
                                                                           model_name, 
                                                                           cache_dir='./')
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
        
    count, all = 0, 0
    wrong_id_list = list()
    try:
        for image, cap1, cap2, label in tqdm(foil_dataset): 
            label = 'A' if label == 0 else 'B'
            qs = f'Which of the following captions better fits the image: \n (A){cap1} \n (B){cap2}'
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
            # if outputs.lower() != label.lower():
            #     wrong_id_list.append(id)
            count += (outputs.lower() == label.lower())
            all += 1
            print(f'count: {count}, all: {all}, acc:{round(count/all, 4)}, output: {outputs}, answer: {label}')
    except Exception as e:
        print(e)
    # import pickle
    # with open(f"hard_samples_{args.model_path}", "wb") as fp:   #Pickling
    #     pickle.dump(wrong_id_list, fp)
        
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

    
