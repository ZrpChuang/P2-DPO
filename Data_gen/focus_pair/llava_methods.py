import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *


NUM_IMG_TOKENS = 576
NUM_PATCHES = 24
PATCH_SIZE = 14
IMAGE_RESOLUTION = 336
IMAGE_TOKEN_INDEX = 32000
ATT_LAYER = 14

import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_attention_llava_all(image, prompt, general_prompt, model, processor):


    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)


    outputs = model(**inputs, output_attentions=True)
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)


    all_attentions = outputs.attentions
    print(f"模型总共输出了 {len(all_attentions)} 个可用的注意力层 (索引从 0 到 {len(all_attentions) - 1})。")


    list_of_att_maps = []


    for i, attention in enumerate(all_attentions):


        grads = torch.autograd.grad(loss, attention, retain_graph=True)
        grad_att = attention * F.relu(grads[0])


        att_map = grad_att[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)


        list_of_att_maps.append(att_map)


    model.zero_grad()

    print(f"处理完成，返回了包含 {len(list_of_att_maps)} 个注意力图的列表。")
    return list_of_att_maps


def gradient_attention_llava(image, prompt, general_prompt, model, processor):


    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)


    outputs = model(**inputs, output_attentions=True)
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)


    attention = outputs.attentions[ATT_LAYER]
    print(f"模型总共输出了 {len(outputs.attentions)} 个可用的注意力层 (索引从 0 到 {len(outputs.attentions) - 1})。")
    grads = torch.autograd.grad(loss, attention, retain_graph=True)
    grad_att = attention * F.relu(grads[0])


    att_map = grad_att[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    model.zero_grad()

    return att_map

def rel_attention_llava(image, prompt, general_prompt, model, processor):


    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)


    att_map = model(**inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)


    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_pos = general_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)


    general_att_map = model(**general_inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, general_pos:general_pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    att_map = att_map / general_att_map

    return att_map

def pure_gradient_llava(image, prompt, general_prompt, model, processor):


    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)


    high_pass = high_pass_filter(image, IMAGE_RESOLUTION, reduce=False)


    inputs['pixel_values'].requires_grad = True
    general_inputs['pixel_values'].requires_grad = True


    criterion = nn.CrossEntropyLoss()


    zero_logit = model(**inputs, output_hidden_states=False).logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -criterion(zero_logit, true_class)


    grads = torch.autograd.grad(loss, inputs['pixel_values'], retain_graph=True)[0]


    general_zero_logit = model(**general_inputs, output_hidden_states=False).logits[:, -1, :]
    general_true_class = torch.argmax(general_zero_logit, dim=1)
    general_loss = -criterion(general_zero_logit, general_true_class)


    general_grads = torch.autograd.grad(general_loss, general_inputs['pixel_values'], retain_graph=True)[0]


    grads = grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    general_grads = general_grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)


    grad = np.linalg.norm(grads, axis=2)
    general_grad = np.linalg.norm(general_grads, axis=2)


    grad = grad / general_grad
    high_pass = high_pass > np.median(high_pass)
    grad = grad * high_pass


    grad = block_reduce(grad, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)

    return grad
