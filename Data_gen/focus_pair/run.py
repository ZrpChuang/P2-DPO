import os
from PIL import Image
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

def vicrop_qa(model_name, method_name, image_path, question, model, processor, short_question,bbox_vicrop):
    ori_avg_log_prob, ori_perplexity = None, None
    multi_avg_log_prob, multi_perplexity = None, None
    entropy, peak_count = None, None
    bbox = None
    ori_generation, multi_generation = "", ""

    image = Image.open(image_path).convert("RGB")
    model.eval()

    general_question = 'Write a general description of the image.'

    short_prompt = f"<image>\nUSER: {short_question}\nASSISTANT:"
    prompt = f"<image>\nUSER:{question}\nASSISTANT:"
    general_prompt = f"<image>\nUSER: {general_question}\nASSISTANT:"
    
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150, 
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True
    )

    ori_generate_ids = outputs.sequences
    transition_scores = outputs.scores

    ori_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(ori_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]

    if len(transition_scores) > 0:
        generated_scores = model.compute_transition_scores(
            ori_generate_ids, transition_scores, normalize_logits=True
        )
        
        ori_avg_log_prob = torch.mean(generated_scores[0]).item()
        ori_perplexity = torch.exp(-torch.mean(generated_scores[0])).item()

    else:
        pass
    
    del inputs
    torch.cuda.empty_cache()

    if method_name == 'grad_att':
        att_map = gradient_attention_llava(image, short_prompt, general_prompt, model, processor)
        
        try:
            base_image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_filename = f"{base_image_name}_{model_name}_{method_name}_heatmap.png"
            
            output_dir = '/data/ruipeng.zhang/dpo_on/test_image_fix_fix'
            save_path = os.path.join(output_dir, save_filename)
            
            save_attention_heatmap(att_map, image, save_path)
            
        except Exception as e:
            pass

        bbox = bbox_from_att_image_adaptive_org(att_map, image.size, bbox_size)
        
        attention_metrics = analyze_attention_map(att_map)
        
        entropy = attention_metrics.get('entropy')
        peak_count = attention_metrics.get('peak_count')
    
    if bbox is None:
        pass
    else:
        crop_image = image.crop(bbox)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = '/data/ruipeng.zhang/dpo_on/test_image_fix'
        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, f"{base_filename}_original.jpg"), "JPEG")
        crop_image.save(os.path.join(output_dir, f"{base_filename}_{method_name}_cropped.jpg"), "JPEG")
     
        multi_prompt = f"""<image><image>\nUSER: Answer the question mainly based on the first image, refer to the cropping details of the second image if necessary,answer this question: {question}\nASSISTANT:"""
        
        multi_inputs = processor(multi_prompt, [image, crop_image], return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        outputs_multi = model.generate(
            **multi_inputs,
            max_new_tokens=150,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

        multi_generate_ids = outputs_multi.sequences
        transition_scores_multi = outputs_multi.scores

        multi_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(multi_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]

        if len(transition_scores_multi) > 0:
            generated_scores_multi = model.compute_transition_scores(
                multi_generate_ids, transition_scores_multi, normalize_logits=True
            )

            multi_avg_log_prob = torch.mean(generated_scores_multi[0]).item()
            multi_perplexity = torch.exp(-torch.mean(generated_scores_multi[0])).item()

        else:
            pass
    
    results = {
        "ori_generation": ori_generation,
        "multi_generation": multi_generation,
        "bbox": bbox,
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
            }
        }
    }
    return results

MODEL_CACHE = {}