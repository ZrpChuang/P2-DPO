import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *

# hyperparameters
NUM_IMG_TOKENS = 576
NUM_PATCHES = 24
PATCH_SIZE = 14
IMAGE_RESOLUTION = 336
IMAGE_TOKEN_INDEX = 32000
ATT_LAYER = 14

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设这些常量在您的代码上下文中已经定义好了
# ATT_LAYER = ...  (这个将不再使用)
# IMAGE_TOKEN_INDEX = -200
# NUM_IMG_TOKENS = 576
# NUM_PATCHES = 24

# --- 新函数：gradient_attention_llava_all ---
def gradient_attention_llava_all(image, prompt, general_prompt, model, processor):
    """
    计算并返回LLaVA模型中所有可用注意力层的梯度注意力图列表。
    函数签名和核心逻辑与原版保持一致，仅修改为返回所有层。
    """
    
    # 准备输入 (完全照抄您的代码)
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)
    
    # 计算损失 (完全照抄您的代码)
    outputs = model(**inputs, output_attentions=True)
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)
    
    # --- 核心修改部分 ---

    # 获取所有层的注意力张量
    all_attentions = outputs.attentions
    print(f"模型总共输出了 {len(all_attentions)} 个可用的注意力层 (索引从 0 到 {len(all_attentions) - 1})。")

    # 用于存储所有层注意力图的列表
    list_of_att_maps = []
    
    # 遍历所有注意力层
    for i, attention in enumerate(all_attentions):
        # 对当前层的注意力计算梯度
        # 注意：每次循环都计算一次梯度，虽然效率稍低，但能确保与原逻辑最接近
        # 且能处理不同层可能有不同计算图的边缘情况
        grads = torch.autograd.grad(loss, attention, retain_graph=True)
        grad_att = attention * F.relu(grads[0])
        
        # 计算注意力图 (完全照抄您的代码)
        att_map = grad_att[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
        
        # 将计算出的当前层注意力图添加到列表中
        list_of_att_maps.append(att_map)

    # 清除梯度以避免累积 (完全照抄您的代码)
    model.zero_grad() 
    
    print(f"处理完成，返回了包含 {len(list_of_att_maps)} 个注意力图的列表。")
    return list_of_att_maps


def gradient_attention_llava(image, prompt, general_prompt, model, processor):
    """
    Generates an attention map using gradient-weighted attention from LLaVA model.
    
    This function computes attention maps from the LLaVA model and weights them by their
    gradients with respect to the loss. It focuses on the attention paid to image tokens
    in the final token prediction, highlighting regions relevant to the prompt.
    
    Args:
        image: Input image to analyze
        prompt: Text prompt for which to generate attention
        general_prompt: General text prompt (not directly used in this function)
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the gradient-weighted attention map
    """
    # Prepare inputs
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)
    
    # Compute loss
    outputs = model(**inputs, output_attentions=True)
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)
    
    # Compute attention and gradients
    attention = outputs.attentions[ATT_LAYER]
    print(f"模型总共输出了 {len(outputs.attentions)} 个可用的注意力层 (索引从 0 到 {len(outputs.attentions) - 1})。")
    grads = torch.autograd.grad(loss, attention, retain_graph=True)
    grad_att = attention * F.relu(grads[0])
    
    # Compute the attention maps
    att_map = grad_att[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    
    model.zero_grad() # Clear gradients to avoid accumulation
    
    return att_map

def rel_attention_llava(image, prompt, general_prompt, model, processor):
    """
    Generates a relative attention map by comparing specific prompt attention to general prompt attention.
    
    This function computes attention maps for both a specific prompt and a general prompt in the LLaVA model,
    then calculates their ratio to highlight regions that are uniquely relevant to the specific prompt.
    It focuses on the attention paid to image tokens in the final token prediction.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate attention
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the relative attention map (specific/general)
    """
    # Prepare inputs for the prompt
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute attention map for the 14th layer
    att_map = model(**inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    # Prepare inputs for the general prompt
    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_pos = general_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute general attention map for the 14th layer
    general_att_map = model(**general_inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, general_pos:general_pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    # Normalize attention map
    att_map = att_map / general_att_map

    return att_map

def pure_gradient_llava(image, prompt, general_prompt, model, processor):
    """
    Generates a gradient-based attention map using direct image gradients in LLaVA.
    
    This function computes gradients of the loss with respect to the input image pixels
    for both specific and general prompts. It then calculates their ratio and applies
    a high-pass filter to highlight fine-grained details that are uniquely relevant to the specific prompt.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate gradients
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        grad: A 2D numpy array representing the processed gradient map highlighting
              regions relevant to the specific prompt
    """
    # Process inputs
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    
    # Apply high pass filter
    high_pass = high_pass_filter(image, IMAGE_RESOLUTION, reduce=False)
    
    # Enable gradients
    inputs['pixel_values'].requires_grad = True
    general_inputs['pixel_values'].requires_grad = True
    
    # Initialize loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass for inputs
    zero_logit = model(**inputs, output_hidden_states=False).logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -criterion(zero_logit, true_class)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, inputs['pixel_values'], retain_graph=True)[0]
    
    # Forward pass for general_inputs
    general_zero_logit = model(**general_inputs, output_hidden_states=False).logits[:, -1, :]
    general_true_class = torch.argmax(general_zero_logit, dim=1)
    general_loss = -criterion(general_zero_logit, general_true_class)
    
    # Compute general gradients
    general_grads = torch.autograd.grad(general_loss, general_inputs['pixel_values'], retain_graph=True)[0]
    
    # Process gradients
    grads = grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    general_grads = general_grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # Compute gradient norms
    grad = np.linalg.norm(grads, axis=2)
    general_grad = np.linalg.norm(general_grads, axis=2)
    
    # Normalize and apply high pass filter
    grad = grad / general_grad
    high_pass = high_pass > np.median(high_pass)
    grad = grad * high_pass
    
    # Reduce gradient block size
    grad = block_reduce(grad, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)
    
    return grad