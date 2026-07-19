import os
from PIL import Image, ImageDraw
import torch
import numpy as np
import inspect
from transformers import LlavaForConditionalGeneration
import os
try:
    file_path = inspect.getfile(LlavaForConditionalGeneration)
except TypeError:
    pass

from llava_methods import *
from utils import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from skimage.feature import peak_local_max

def save_attention_heatmap(att_map, original_image, save_path, colormap=plt.cm.jet, alpha=0.6):
    if isinstance(original_image, Image.Image):
        original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    elif isinstance(original_image, np.ndarray):
        original_image_cv = original_image
    else:
        raise TypeError("original_image must be of type PIL.Image or numpy.ndarray")

    h, w, _ = original_image_cv.shape

    att_map_normalized = cv2.normalize(att_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    heatmap_resized = cv2.resize(att_map_normalized, (w, h), interpolation=cv2.INTER_CUBIC)

    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(colored_heatmap, alpha, original_image_cv, 1 - alpha, 0)

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    cv2.imwrite(save_path, superimposed_img)

def analyze_attention_map(att_map: np.ndarray, min_distance: int = 5, threshold_rel: float = 0.3) -> dict:
    metrics = {
        'entropy': 0.0,
        'peak_count': 0,
        'peak_coordinates': [],
        'peak_heights': [],
        'peak_to_average_ratio': 0.0,
        'standard_deviation': 0.0,
        'max_attention_value': 0.0
    }

    if not isinstance(att_map, np.ndarray) or att_map.sum() == 0:
        return metrics

    prob_map = att_map / att_map.sum()
    non_zero_probs = prob_map[prob_map > 0]
    metrics['entropy'] = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    threshold_abs = att_map.max() * threshold_rel
    
    coordinates = peak_local_max(
        att_map,
        min_distance=min_distance,
        threshold_abs=threshold_abs
    )
    
    metrics['peak_count'] = len(coordinates)
    metrics['peak_coordinates'] = coordinates.tolist()

    if metrics['peak_count'] > 0:
        peak_heights = [att_map[y, x] for y, x in coordinates]
        metrics['peak_heights'] = peak_heights

    metrics['max_attention_value'] = np.max(att_map)
    metrics['standard_deviation'] = np.std(att_map)
    
    mean_val = np.mean(att_map)
    if mean_val > 0:
        metrics['peak_to_average_ratio'] = np.max(att_map) / mean_val

    return metrics

bbox_size = 336

def _normalize_bbox(bbox):
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(key in bbox for key in keys):
            return tuple(int(round(float(bbox[key]))) for key in keys)
        keys = ("left", "top", "right", "bottom")
        if all(key in bbox for key in keys):
            return tuple(int(round(float(bbox[key]))) for key in keys)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return tuple(int(round(float(value))) for value in bbox[:4])
    return None

def _clip_bbox(bbox, image_size):
    if bbox is None:
        return None
    width, height = image_size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return (x1, y1, x2, y2)

def _erase_region(image, bbox):
    erased = image.copy()
    pixels = np.asarray(image)
    fill = tuple(int(value) for value in pixels.reshape(-1, 3).mean(axis=0))
    draw = ImageDraw.Draw(erased)
    draw.rectangle(bbox, fill=fill)
    return erased

def _decode_generation(processor, generate_ids):
    decoded = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    if "ASSISTANT:" in decoded:
        return decoded.split("ASSISTANT:", 1)[1].strip()
    return decoded.strip()

def _run_generation(model, processor, prompt, images):
    inputs = processor(prompt, images, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True
    )
    generate_ids = outputs.sequences
    text = _decode_generation(processor, generate_ids)
    avg_log_prob, perplexity = None, None
    if len(outputs.scores) > 0:
        generated_scores = model.compute_transition_scores(
            generate_ids,
            outputs.scores,
            normalize_logits=True
        )
        avg_log_prob = torch.mean(generated_scores[0]).item()
        perplexity = torch.exp(-torch.mean(generated_scores[0])).item()
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, avg_log_prob, perplexity

def vicrop_qa(model_name, method_name, image_path, question, model, processor, short_question, bbox_vicrop):
    ori_avg_log_prob, ori_perplexity = None, None
    multi_avg_log_prob, multi_perplexity = None, None
    deg_avg_log_prob, deg_perplexity = None, None
    entropy, peak_count = None, None
    bbox = _normalize_bbox(bbox_vicrop)
    ori_generation, multi_generation, deg_generation = "", "", ""

    image = Image.open(image_path).convert("RGB")
    model.eval()

    general_question = 'Write a general description of the image.'

    short_prompt = f"<image>\nUSER: {short_question}\nASSISTANT:"
    prompt = f"<image>\nUSER:{question}\nASSISTANT:"
    general_prompt = f"<image>\nUSER: {general_question}\nASSISTANT:"
    
    ori_generation, ori_avg_log_prob, ori_perplexity = _run_generation(model, processor, prompt, image)

    if method_name == 'grad_att' and bbox is None:
        att_map = gradient_attention_llava(image, short_prompt, general_prompt, model, processor)
        bbox = bbox_from_att_image_adaptive_org(att_map, image.size, bbox_size)
        attention_metrics = analyze_attention_map(att_map)
        entropy = attention_metrics.get('entropy')
        peak_count = attention_metrics.get('peak_count')
    
    bbox = _clip_bbox(bbox, image.size)
    if bbox is not None:
        crop_image = image.crop(bbox)
        deg_image = _erase_region(image, bbox)
        multi_prompt = f"<image><image>\nUSER: Answer the question mainly based on the first image, refer to the cropping details of the second image if necessary, answer this question: {question}\nASSISTANT:"
        multi_generation, multi_avg_log_prob, multi_perplexity = _run_generation(
            model,
            processor,
            multi_prompt,
            [image, crop_image]
        )
        deg_generation, deg_avg_log_prob, deg_perplexity = _run_generation(
            model,
            processor,
            prompt,
            deg_image
        )
    
    results = {
        "ori_generation": ori_generation,
        "multi_generation": multi_generation,
        "deg_generation": deg_generation,
        "bbox": bbox,
        "focus": {
            "chosen": multi_generation,
            "rejected": deg_generation,
            "bbox": list(bbox) if bbox is not None else None,
            "chosen_logp": multi_avg_log_prob,
            "rejected_logp": deg_avg_log_prob,
            "chosen_ppl": multi_perplexity,
            "rejected_ppl": deg_perplexity
        },
        "metrics": {
            "attention": {
                "entropy": float(entropy) if entropy is not None else None,
                "peak_count": int(peak_count) if peak_count is not None else None
            },
            "original_output": {
                "avg_log_prob": ori_avg_log_prob,
                "perplexity": ori_perplexity
            },
            "multi_image_output": {
                "avg_log_prob": multi_avg_log_prob,
                "perplexity": multi_perplexity
            },
            "degraded_output": {
                "avg_log_prob": deg_avg_log_prob,
                "perplexity": deg_perplexity
            }
        }
    }
    return results

def vicrop_qa_champion(model_name, method_name, image_path, question, model, processor, short_question, bbox_vicrop, chosen_answer=None):
    return vicrop_qa(
        model_name=model_name,
        method_name=method_name,
        image_path=image_path,
        question=question,
        model=model,
        processor=processor,
        short_question=short_question,
        bbox_vicrop=bbox_vicrop
    )

MODEL_CACHE = {}
