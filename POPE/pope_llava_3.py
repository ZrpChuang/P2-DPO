import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava_dpo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_dpo.conversation import conv_templates, SeparatorStyle
from llava_dpo.model.builder import load_pretrained_model
from llava_dpo.utils import disable_torch_init
from llava_dpo.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_dpo.model import *
from PIL import Image
import math

from transformers import set_seed


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
        with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    
                    do_sample=False,
                    num_beams=2,
                    
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image-folder", type=str, default="data/coco/val2014")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--question-dir", type=str, default="data/POPE/coco")
    parser.add_argument("--answers-dir", type=str, default="outputs/POPE")

    args = parser.parse_args()

    pope_modes = ['adversarial', 'random', 'popular']
    base_question_path = args.question_dir
    base_answers_path = args.answers_dir

    for mode in pope_modes:
        print(f"--- [Starting mode: {mode}] ---")

        question_file = os.path.join(base_question_path, f"coco_pope_{mode}.json")
        answers_file = os.path.join(base_answers_path, f"coco_{mode}_nb2.jsonl")
        
        args.question_file = question_file
        args.answers_file = answers_file

        output_dir = os.path.dirname(args.answers_file)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Question File: {args.question_file}")
        print(f"Answers File: {args.answers_file}")

        set_seed(args.seed)
        eval_model(args)
        
        print(f"--- [Mode: {mode} finished] ---\n")

    print("All modes processed!")
