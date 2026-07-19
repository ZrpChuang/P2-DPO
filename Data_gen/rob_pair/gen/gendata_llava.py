import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import copy
from transformers import GenerationConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling

evolve_vcd_sampling()


def decode_and_clean_outputs(output_ids, input_token_len, tokenizer, stop_str):
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(os.path.expanduser(args.question_file), "r") as f:
        full_data = json.load(f)
    
    results_data = []

    ct = 1
    for item in tqdm(full_data):
        ct += 1
        new_item = copy.deepcopy(item)

        human_conv = next((conv for conv in item["conversations"] if conv["from"] == "human"), None)
        if not human_conv:
            print(f"Skipping item with idx {item.get('idx', 'N/A')} as no human conversation was found.")
            continue

        image_file = item["image"]
        raw_qs = human_conv["value"]
        if raw_qs.startswith(DEFAULT_IMAGE_TOKEN):
            qs = raw_qs.replace(DEFAULT_IMAGE_TOKEN, '', 1).lstrip('\n')
        else:
            qs = raw_qs
        prompt_prefix_factual = "Strictly answer the following question according to the facts of the image, and control the length of the output. Based on this, answer the question:"
        final_text_factual = f"{prompt_prefix_factual} {qs}"
        if model.config.mm_use_im_start_end:
            full_qs_factual = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + final_text_factual
        else:
            full_qs_factual = DEFAULT_IMAGE_TOKEN + '\n' + final_text_factual

        conv_factual = conv_templates[args.conv_mode].copy()
        conv_factual.append_message(conv_factual.roles[0], full_qs_factual)
        conv_factual.append_message(conv_factual.roles[1], None)
        prompt_factual = conv_factual.get_prompt()
        input_ids_factual = tokenizer_image_token(prompt_factual, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').squeeze(0)

        try:
            image = Image.open(os.path.join(args.image_folder, image_file))
        except FileNotFoundError:
            print(f"Image not found, skipping: {os.path.join(args.image_folder, image_file)}")
            continue

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step) if args.use_cd else None


        with torch.inference_mode():
            generation_config_normal = GenerationConfig.from_model_config(model.config)
            generation_config_normal.do_sample = True
            generation_config_normal.temperature = args.temperature
            generation_config_normal.top_p = args.top_p
            generation_config_normal.max_new_tokens = 256
            generation_config_normal.generation_mode = 'normal'
            if args.top_k is not None:
                generation_config_normal.top_k = args.top_k

            output_ids_normal = model.generate(
                input_ids_factual.unsqueeze(0).cuda(), 
                images=image_tensor.unsqueeze(0).half().cuda(),
                generation_config=generation_config_normal
            )
        new_item["normal_answer"] = decode_and_clean_outputs(output_ids_normal, input_ids_factual.shape[0], tokenizer, "</s>")


        if args.use_cd and image_tensor_cd is not None:
            
            with torch.inference_mode():
                generation_config_noisy = GenerationConfig.from_model_config(model.config)
                generation_config_noisy.do_sample = True
                generation_config_noisy.temperature = args.temperature
                generation_config_noisy.top_p = args.top_p
                generation_config_noisy.max_new_tokens = 256
                generation_config_noisy.generation_mode = 'noisy'
                if args.top_k is not None:
                    generation_config_noisy.top_k = args.top_k
                
                output_ids_noisy = model.generate(
                    input_ids_factual.unsqueeze(0).cuda(),
                    images=image_tensor.unsqueeze(0).half().cuda(), 
                    images_cd=image_tensor_cd.unsqueeze(0).half().cuda(),
                    generation_config=generation_config_noisy
                )
            new_item["noisy_answer"] = decode_and_clean_outputs(output_ids_noisy, input_ids_factual.shape[0], tokenizer, "</s>")
            
            with torch.inference_mode():
                generation_config_cd = GenerationConfig.from_model_config(model.config)
                generation_config_cd.do_sample = True
                generation_config_cd.temperature = args.temperature
                generation_config_cd.top_p = args.top_p
                generation_config_cd.max_new_tokens = 256
                generation_config_cd.generation_mode = 'contrastive'
                generation_config_cd.cd_alpha = args.cd_alpha
                generation_config_cd.cd_beta = args.cd_beta
                if args.top_k is not None:
                    generation_config_cd.top_k = args.top_k
                
                output_ids_cd = model.generate(
                    input_ids_factual.unsqueeze(0).cuda(), 
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=image_tensor_cd.unsqueeze(0).half().cuda(),
                    generation_config=generation_config_cd
                )
            new_item["cd_answer"] = decode_and_clean_outputs(output_ids_cd, input_ids_factual.shape[0], tokenizer, "</s>")
            new_item["robust"] = {
                "chosen": new_item["cd_answer"],
                "rejected": new_item["noisy_answer"],
                "noise_step": args.noise_step,
                "cd_alpha": args.cd_alpha,
                "cd_beta": args.cd_beta
            }
        else:
            new_item["noisy_answer"] = "N/A (use_cd is False)"
            new_item["cd_answer"] = "N/A (use_cd is False)"
            new_item["robust"] = {
                "chosen": new_item["cd_answer"],
                "rejected": new_item["noisy_answer"],
                "noise_step": args.noise_step,
                "cd_alpha": args.cd_alpha,
                "cd_beta": args.cd_beta
            }

        results_data.append(new_item)

        

    with open(os.path.expanduser(args.answers_file), "w") as ans_file:
        json.dump(results_data, ans_file, indent=2)

    print(f"Processing complete. Results saved to {args.answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None) 
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--noise_step", type=int, default=600)
    parser.add_argument("--use_cd", action='store_true', default=True)
    parser.add_argument("--cd_alpha", type=float, default=0.1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for sampling. Usually set to None if top_p is used.")
    
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
