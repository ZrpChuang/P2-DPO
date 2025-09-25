import json
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from PIL import Image

try:
    from run import vicrop_qa,vicrop_qa_champion
except ImportError:
    print("Error: Cannot import 'vicrop_qa' function. Please ensure 'run.py' file exists and is in the correct path.")
    exit()

INPUT_JSON_PATH = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_detailed_description.json'
OUTPUT_JSON_PATH = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_detailed_description_ans_clip.json'
IMAGE_BASE_PATH = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_images'

MODEL_NAME = 'llava'
METHOD_NAME = 'grad_att'

print("Loading LLaVA model and processor...")
try:
    model = LlavaForConditionalGeneration.from_pretrained(
        'llava-hf/llava-1.5-7b-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to('cuda')
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error occurred while loading model or processor: {e}")
    exit()

print(f"Reading data from {INPUT_JSON_PATH}...")
try:
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_JSON_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: File {INPUT_JSON_PATH} is not a valid JSON file.")
    exit()

processed_results = []
print(f"Starting to process {len(data)} items...")

for item in tqdm(data, desc="Processing JSON items"):
    relative_image_path = item.get('image')
    if not relative_image_path:
        print(f"\nWarning: Item {item.get('idx', 'N/A')} is missing 'image' field, skipping.")
        item['error'] = "Missing 'image' field"
        processed_results.append(item)
        continue
        
    full_image_path = os.path.join(IMAGE_BASE_PATH, relative_image_path)

    if not os.path.exists(full_image_path):
        print(f"\nWarning: Image file not found at '{full_image_path}', skipping item {item.get('idx', 'N/A')}.")
        item['error'] = f"Image file not found at {full_image_path}"
        processed_results.append(item)
        continue

    raw_question = item['conversations'][0]['value']
    question = raw_question.replace('<image>\n', '').strip()
    short_question = question
    chosen_answer = item['conversations'][1]['value']

    bbox_vicrop = item.get('bbox_vicrop')
    results = vicrop_qa_champion(
        model_name=MODEL_NAME,
        method_name=METHOD_NAME,
        image_path=full_image_path,
        question=question,
        model=model,
        processor=processor,
        short_question=short_question,
        bbox_vicrop=bbox_vicrop,
        chosen_answer=chosen_answer
    )

    if results:
        ori_answer = results.get('ori_generation', '')
        crop_answer = results.get('multi_generation', '')
        bbox = results.get('bbox')
        
        item['ori_answer_vicrop'] = ori_answer
        item['crop_answer_vicrop'] = crop_answer
        
        item['metrics_vicrop'] = results.get('metrics', {})
        
    else:
        print(f"\nvicrop_qa function failed and returned None while processing item {item.get('idx', 'N/A')}.")
        item['error'] = 'vicrop_qa function failed and returned None.'

    processed_results.append(item)

print(f"\nProcessing complete. Saving results to {OUTPUT_JSON_PATH} ...")
try:
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    print("All results have been saved successfully!")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")