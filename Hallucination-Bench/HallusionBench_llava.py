import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from llava_dpo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava_dpo.conversation import conv_templates, SeparatorStyle
    from llava_dpo.model.builder import load_pretrained_model
    from llava_dpo.utils import disable_torch_init
    from llava_dpo.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
except ImportError as e:
    print("Error: Failed to import LLaVA modules. Please ensure your Python environment is set up correctly,")
    print("and the current working directory or PYTHONPATH includes your llava_dpo project root.")
    print(f"Specific error: {e}")
    sys.exit(1)


def generate_mmhal_responses(args):
    """
    Loads the MMHal-Bench dataset, generates responses using your trained LLaVA model,
    and saves them to a new JSON file.
    """
    print("Loading model...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    print(f"Model '{model_name}' loaded successfully.")

    template_file_path = os.path.expanduser(args.template_file)
    print(f"Loading question templates from {template_file_path}...")
    with open(template_file_path, "r", encoding='utf-8') as f:
        bench_data = json.load(f)
    print(f"Loaded {len(bench_data)} evaluation samples.")

    results_list = []

    for item in tqdm(bench_data, desc="Evaluating MMHal-Bench"):
        question_text = item["question"]
        
        image_filename = item["image_src"].split('/')[-1]
        image_path = os.path.join(args.image_folder, image_filename)
        
        if model.config.mm_use_im_start_end:
            prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_text
        else:
            prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + question_text

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        image_tensor = None
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor = image_tensor.unsqueeze(0).half().cuda()
        except FileNotFoundError:
            print(f"\n[Warning] Image file not found: {image_path}. Skipping this sample.")
            item['model_answer'] = f"ERROR: Image not found at {image_path}"
            results_list.append(item)
            continue
        except Exception as e:
            print(f"\n[Error] Failed to load or process image: {image_path}. Error: {e}. Skipping this sample.")
            item['model_answer'] = f"ERROR: Could not process image. {e}"
            results_list.append(item)
            continue

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            generate_kwargs = {
                "input_ids": input_ids,
                "images": image_tensor,
                "do_sample": False,
                "num_beams": args.num_beams,
                "max_new_tokens": 1024,
                "use_cache": True,
                "stopping_criteria": [stopping_criteria]
            }
            output_ids = model.generate(**generate_kwargs)

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        model_answer = outputs.strip()

        item['model_answer'] = model_answer
        results_list.append(item)

    output_file_path = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete! Results have been saved to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-path", type=str, default="", help="Path to your model weights.")
    parser.add_argument("--model-base", type=str, default="llava-hf/llava-1.5-7b-hf", help="Path to the base model if your model is a LoRA adapter.")
    parser.add_argument("--template-file", type=str, default="data/MMHal-Bench/response_template.json", help="Path to the MMHal-Bench template file.")
    parser.add_argument("--image-folder", type=str, default="data/MMHal-Bench/images", help="Path to the folder containing all MMHal-Bench images.")
    parser.add_argument("--output-file", type=str, default="./output/r32_mix_wt_on.json", help="Path to the output file for evaluation results.")
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation template mode.")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search.")
    
    args = parser.parse_args()
    
    generate_mmhal_responses(args)
