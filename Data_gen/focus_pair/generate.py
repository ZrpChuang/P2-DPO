import argparse
import json
import os

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

try:
    from run import vicrop_qa_champion
except ImportError:
    print("Error: Cannot import 'vicrop_qa_champion'. Please ensure 'run.py' is in the current path.")
    raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--image-base-path", required=True)
    parser.add_argument("--model-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="llava")
    parser.add_argument("--method-name", default="grad_att")
    return parser.parse_args()


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path, device):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def extract_question(item):
    raw_question = item["conversations"][0]["value"]
    return raw_question.replace("<image>\n", "").strip()


def main():
    args = parse_args()

    print("Loading LLaVA model and processor...")
    try:
        model, processor = load_model(args.model_path, args.device)
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error occurred while loading model or processor: {e}")
        raise SystemExit(1)

    print(f"Reading data from {args.input_json}...")
    try:
        data = load_data(args.input_json)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_json}")
        raise SystemExit(1)
    except json.JSONDecodeError:
        print(f"Error: File {args.input_json} is not a valid JSON file.")
        raise SystemExit(1)

    processed_results = []
    print(f"Starting to process {len(data)} items...")

    for item in tqdm(data, desc="Processing JSON items"):
        relative_image_path = item.get("image")
        if not relative_image_path:
            print(f"\nWarning: Item {item.get('idx', 'N/A')} is missing 'image' field, skipping.")
            item["error"] = "Missing 'image' field"
            processed_results.append(item)
            continue

        full_image_path = os.path.join(args.image_base_path, relative_image_path)
        if not os.path.exists(full_image_path):
            print(f"\nWarning: Image file not found at '{full_image_path}', skipping item {item.get('idx', 'N/A')}.")
            item["error"] = f"Image file not found at {full_image_path}"
            processed_results.append(item)
            continue

        question = extract_question(item)
        bbox_vicrop = item.get("bbox_vicrop")
        results = vicrop_qa_champion(
            model_name=args.model_name,
            method_name=args.method_name,
            image_path=full_image_path,
            question=question,
            model=model,
            processor=processor,
            short_question=question,
            bbox_vicrop=bbox_vicrop
        )

        if results:
            item["ori_answer_vicrop"] = results.get("ori_generation", "")
            item["crop_answer_vicrop"] = results.get("multi_generation", "")
            item["deg_answer_vicrop"] = results.get("deg_generation", "")
            item["focus"] = results.get("focus")
            item["metrics_vicrop"] = results.get("metrics", {})
        else:
            print(f"\nvicrop_qa function failed and returned None while processing item {item.get('idx', 'N/A')}.")
            item["error"] = "vicrop_qa function failed and returned None."

        processed_results.append(item)

    print(f"\nProcessing complete. Saving results to {args.output_json} ...")
    try:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)
        print("All results have been saved successfully!")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
