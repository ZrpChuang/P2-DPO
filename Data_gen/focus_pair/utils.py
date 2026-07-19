import torchvision.transforms.functional as TF
import numpy as np
from scipy.ndimage import median_filter
from skimage.measure import block_reduce
from io import BytesIO
import base64

def encode_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):
    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_shape):
    if not (isinstance(bbox_shape, tuple) and len(bbox_shape) == 2):
        raise ValueError(f"bbox_shape must be a (width, height) tuple, but got {bbox_shape}")

    base_width, base_height = bbox_shape

    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        target_width = base_width * ratio
        target_height = base_height * ratio

        block_size_w = image_size[0] / att_map.shape[1]
        block_size_h = image_size[1] / att_map.shape[0]

        block_num_w = min(int(target_width / block_size_w), att_map.shape[1])
        block_num_h = min(int(target_height / block_size_h), att_map.shape[0])

        if att_map.shape[1] - block_num_w < 1 and att_map.shape[0] - block_num_h < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue

        block_num = (block_num_w, block_num_h)
        block_nums.append(block_num)

        sliding_att = np.zeros((att_map.shape[0] - block_num_h + 1, att_map.shape[1] - block_num_w + 1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        for x in range(att_map.shape[1] - block_num_w + 1):
            for y in range(att_map.shape[0] - block_num_h + 1):
                att = att_map[y:y+block_num_h, x:x+block_num_w].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)

        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])

        difference = (max_att - np.mean(adjcent_atts)) / (block_num_w * block_num_h)
        differences.append(difference)
        max_att_poses.append(max_att_pos)

    best_index = np.argmax(differences)
    max_att_pos = max_att_poses[best_index]
    block_num = block_nums[best_index]

    selected_bbox_width = base_width * ratios[best_index]
    selected_bbox_height = base_height * ratios[best_index]

    x_center = int(max_att_pos[0] * block_size_w + block_size_w * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size_h + block_size_h * block_num[1] / 2)

    x_center = selected_bbox_width//2 if x_center < selected_bbox_width//2 else x_center
    y_center = selected_bbox_height//2 if y_center < selected_bbox_height//2 else y_center
    x_center = image_size[0] - selected_bbox_width//2 if x_center > image_size[0] - selected_bbox_width//2 else x_center
    y_center = image_size[1] - selected_bbox_height//2 if y_center > image_size[1] - selected_bbox_height//2 else y_center

    x1 = int(max(0, x_center - selected_bbox_width//2))
    y1 = int(max(0, y_center - selected_bbox_height//2))
    x2 = int(min(image_size[0], x_center + selected_bbox_width//2))
    y2 = int(min(image_size[1], y_center + selected_bbox_height//2))

    return x1, y1, x2, y2


def bbox_from_att_image_adaptive_org(att_map, image_size, bbox_size=336):
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))

        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        for x in range(att_map.shape[1]-block_num[0]+1):
            for y in range(att_map.shape[0]-block_num[1]+1):
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)

        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]

    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)

    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=1024):
    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))

    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)

    return split_images, vertical_split, horizontal_split

def high_res(map_func, image, prompt, general_prompt, model, processor):
    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image)
    att_maps = []
    for split_image in split_images:
        att_map = map_func(split_image, prompt, general_prompt, model, processor)
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])

    return block_att
