import argparse
import json
import math
import os
import numpy as np
import ast
import pandas as pd
from os.path import join
import torch
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
all_options = ['A', 'B', 'C', 'D']

def plot_histogram(data_list, label_list, save_path, color_list=["blue", "green", "#008080", "#FCA592", "yellow", "lightblue"],
                   y_bins_max=70, y_bins_slot=10, x_bins_max=0.5, x_bins_slot=0.01, label_size=20):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set(style="darkgrid")
    bins = np.arange(0, x_bins_max, x_bins_slot)
    ybins = np.arange(0, y_bins_max, y_bins_slot)
    plt.rcParams['font.size'] = 2

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, data in enumerate(data_list):
        sns.histplot(data=data, color=color_list[idx], label=label_list[idx], kde=True, bins=100)

    plt.xlabel("")
    plt.ylabel("")
    plt.legend(fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=label_size)
    # ax.set_xticks(bins)
    # ax.set_yticks(ybins)
    # ax.set(xlim=(0, x_bins_max), ylim=(0, y_bins_max))
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.show() 

def plot_heatmap(tokens, original_image, save_path):
    import cv2   
    import numpy as np
    import matplotlib.pyplot as plt
    importance_scores = np.linalg.norm(tokens.float(), axis=1)
    original_image = original_image.to(torch.uint8)
    n = tokens.shape[0]
    # Determine the grid size (h, w)
    h = int(np.sqrt(n))
    w = h  # Assuming square grid for simplicity
    original_image = original_image.permute(1, 2, 0)
    # Reshape importance scores to grid
    token_grid = importance_scores.reshape(h, w)

    # Resize heatmap to match the original image size
    heatmap = cv2.resize(token_grid, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize heatmap
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Apply color map
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)
    original_image = original_image.cpu().numpy().astype(np.uint8)

# Step 2: Convert to np.uint8
    # Convert original image to RGB if it's in BGR
    if original_image.shape[2] == 3:
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_image_rgb = original_image

    # Overlay heatmap onto the original image
    alpha = 0.6
    # original_image_rgb = original_image_rgb.astype(np.uint8)
    overlay = cv2.addWeighted(original_image_rgb, alpha, heatmap_color, 1 - alpha, 0)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.show()    

def load_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name) 

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    return tokenizer, model, image_processor, context_len
    
def calculate_pairwise_mi(vis_tokens, text_tokens):
    import time
    from sklearn.feature_selection import mutual_info_regression

    pairwise_mi = []
    start_time = time.time()
    for vis in vis_tokens:
        for te in text_tokens:
            mi = mutual_info_regression(vis.reshape(-1, 1), te)
            pairwise_mi.append(mi[0])  
            #print(f'index: {i},{j} mi: {mi[0]}')# mutual_info_regression returns a 1D array

    return pairwise_mi

def eval_model(args):
    print('Evaluation Started')
        
    question = 'What is the man wearing?'
    image = [Image.open('/coco/val2014/COCO_val2014_000000006712.jpg').convert('RGB'), 
                Image.open('crop.jpg').convert('RGB')]
        
    qs = cur_prompt = question
    tokenizer, model, image_processor, context_len = load_model(args)

    prompt_qs = qs.split('\n')[0]
    prompt_input_ids = model.get_vision_tower().tokenizer(prompt_qs) 
    
    image_tensor = process_images(image, image_processor, model.config)
    print(image_tensor.shape)
    previous_tokens = model.get_vision_tower().vision_tower(image_tensor.half().cuda(), 
                                                            output_hidden_states=True)
    crop_token = previous_tokens.hidden_states[-2][0, 0].detach().clone().cpu()
    previous_tokens = previous_tokens.hidden_states[-2][1:, 1:].detach().clone().cpu()
    
    current_tokens = model.get_vision_tower()(image_tensor[1:].half().cuda(), 
                                                prompt_input_ids.cuda()).detach().clone().cpu()

    
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    cos_list = [[], []]
    norm_list = [[], []]
    for idx in range(previous_tokens.shape[1]):
        cos_list[0].append(cos_sim(previous_tokens[0][idx].unsqueeze(0), crop_token.unsqueeze(0)).item())
        cos_list[1].append(cos_sim(current_tokens[0][idx].unsqueeze(0), crop_token.unsqueeze(0)).item())
        
        norm_list[0].append(torch.norm(previous_tokens[0][idx].unsqueeze(0)- crop_token.unsqueeze(0), dim=1, p =2).item())
        norm_list[1].append(torch.norm(current_tokens[0][idx].unsqueeze(0)- crop_token.unsqueeze(0), dim=1, p =2).item())
        
    print('llava', np.max(cos_list[0]), np.max(np.abs(cos_list[0])), np.max(norm_list[0]))
    print('emma', np.max(cos_list[1]), np.max(np.abs(cos_list[1])), np.max(norm_list[1]))
        
    #     cos_list[0].append(calculate_pairwise_mi(answer[0][:1].detach().numpy(), 
    #                                              previous_tokens[0].detach().numpy()))
    #     cos_list[1].append(calculate_pairwise_mi(answer[0][:1].detach().numpy(), 
    #                                              current_tokens[0].detach().numpy()))
    #     # cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    #     # cos_list = [[], []]
    #     # for i in range(current_tokens.shape[1] - 1):
    #     #     cos_list[1].append(max(cos_sim(answer[0][:1], current_tokens[0][i:i+1])).item())
    #     #     cos_list[0].append(max(cos_sim(answer[0][:1], previous_tokens[0][i:i+1])).item())
    #     idx += 1
    #     # if idx >= 100:
       
    # plot_histogram(cos_list, ['LLaVA-1.5', 'EMMA-1.0'], save_path='cos.pdf')
    # import pickle
    
    # with open('mi', 'wb') as file:
    #     pickle.dump(cos_list, file)     
            

        
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         inputs=input_ids,
        #         images=image_tensor.half().cuda(),
        #         image_sizes=[image.size],
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         # no_repeat_ngram_size=3,
        #         max_new_tokens=1024,
        #         use_cache=True,
        #         prompt_input_ids=prompt_input_ids)

        # print(row['image_id'], row['question_id'], annotaions_dict[row['image_id']])
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # answer = annotaions_dict[row['image_id']][row['question_id']]
        # count += (outputs.lower() in answer)
        # all += 1
        # print(f'count: {count}, all: {all}, acc: {round(count/all, 4)} outputs: {outputs}, answer: {answer}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/emma-7b-v1-1.8M")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/vast/sg7457/eval/llava-bench-in-the-wild/images")
    parser.add_argument("--question-file", type=str, default="/vast/sg7457/eval/llava-bench-in-the-wild/questions.jsonl")
    parser.add_argument("--annotation-file", type=str, default="/vast/sg7457/eval/llava-bench-in-the-wild/answers_gpt4.jsonl")
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
