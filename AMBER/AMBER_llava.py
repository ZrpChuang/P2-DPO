import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava_dpo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_dpo.conversation import conv_templates, SeparatorStyle
from llava_dpo.model.builder import load_pretrained_model
from llava_dpo.utils import disable_torch_init
from llava_dpo.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def generate_model_responses(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    output_file_path = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w", encoding='utf-8') as ans_file:
        for item in tqdm(questions, desc="Generating responses"):
            item_id = item["id"]
            image_file = item["image"]
            query_text = item["query"]

            if model.config.mm_use_im_start_end:
                prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query_text
            else:
                prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + query_text

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            try:
                image_path = os.path.join(args.image_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            except FileNotFoundError:
                continue

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            response_dict = {
                "id": item_id,
                "response": outputs
            }
            ans_file.write(json.dumps(response_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--model-base", type=str,  default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--question-file", type=str, default="data/AMBER/query/query_all.json")
    parser.add_argument("--image-folder", type=str, default="data/AMBER/images")
    parser.add_argument("--output-file", type=str, default="outputs/AMBER/AMBER_llava_responses.jsonl")

    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation template mode.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation; smaller is more deterministic.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (nucleus) sampling parameter.")

    args = parser.parse_args()

    generate_model_responses(args)
