import argparse
import json
import math
import os

import shortuuid
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, question_tokenizer):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.question_tokenizer = question_tokenizer
        count = 0
        self.idx_list = list()
        for idx, line in enumerate(self.questions):
            image_file = line["image"]
            try:
                image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
                self.idx_list.append(idx)
            except Exception as e:
                count += 1
                print(f'error: {e}, problematic images: {count}')

    def __getitem__(self, index):
        line = self.questions[self.idx_list[index]]
        image_file = line["image"]
        qs = line["text"]
        prompt_qs = qs.split('\n')[0]
        prompt_input_ids = self.question_tokenizer(prompt_qs)
        # prompt_input_ids = self.question_tokenizer('describe this image')
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return prompt_input_ids, input_ids, image, image.size

    def __len__(self):
        return len(self.idx_list)


def collate_fn(batch):
    prompt_input_ids, input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    prompt_input_ids = torch.stack(prompt_input_ids, dim=0)
    return prompt_input_ids, input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4,
                       question_tokenizer=None):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config,
                            question_tokenizer=question_tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             collate_fn=collate_fn)
    return dataset  # data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, 
                                                                           model_name, cache_dir='./')
    normalize = transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config,
                                     question_tokenizer=model.get_vision_tower().tokenizer)

    for (prompt_input_ids, input_ids, image, image_sizes), line in tqdm(zip(data_loader, questions),
                                                                        total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids, prompt_input_ids = input_ids.to(device='cuda', non_blocking=True), prompt_input_ids.to(device='cuda',
                                                                                                          non_blocking=True)
        
        image_tensor = process_images([image], image_processor, model.config)[0]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.unsqueeze(0),
                images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                prompt_input_ids=prompt_input_ids)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--attack", type=str, default='none')
    parser.add_argument('--inner_loss', type=str, default='l2')
    parser.add_argument('--iterations_adv', default=20, type=int, help="number of attack iterations to use")
    args = parser.parse_args()

    eval_model(args)
