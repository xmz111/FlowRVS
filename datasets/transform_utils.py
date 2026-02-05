from os.path import join
from PIL import Image
import datasets.transforms_video as TV

from torch.utils.data import Dataset
import torchvision.transforms as TF
import random
import numpy as np
import torch
from utils_inf import colormap
import os
import math
import torch.nn.functional as F
# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

def resize_to_multiple_of_32_and_fit_box(img, multiple=32, target_h=480, target_w=832):
    """
    Resizes an image to fit within a target_h x target_w box,
    while ensuring the final height and width are multiples of 'multiple'.
    It aims for the largest possible dimensions under these constraints.
    """
    if isinstance(img, torch.Tensor):
        original_height, original_width = img.shape[-2:]
    else: 
        original_width, original_height = img.size

    h_ratio = target_h / original_height
    w_ratio = target_w / original_width
    final_ratio = min(h_ratio, w_ratio)

    final_h = int(original_height * final_ratio)
    final_w = int(original_width * final_ratio)

    final_h = (final_h // multiple) * multiple
    final_w = (final_w // multiple) * multiple

    resized_transform = TF.Resize([final_h, final_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    
    return resized_transform(img)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.25 + color * 0.75
    origin_img = Image.fromarray(origin_img)
    return origin_img

def vis_add_mask_new(img, mask, color, alpha: float = 0.5):
    """
    Overlays a mask on an image with adjustable transparency.

    Args:
        img (Image.Image): The original PIL Image.
        mask (np.ndarray): The binary mask (0 or 1). Expected shape: (H, W).
        color (List[int]): The RGB color for the mask (e.g., [255, 0, 0] for red).
        alpha (float): The transparency level for the mask (0.0 fully transparent, 1.0 fully opaque).
                         Defaults to 0.7.

    Returns:
        Image.Image: The image with the mask overlaid.
    """
    # Ensure mask is binary and has the correct shape for blending
    color = (255, 0, 255)
    mask_np = np.asarray(mask).astype(np.uint8)
    if mask_np.ndim > 2:
        mask_np = mask_np.reshape(mask_np.shape[0], mask_np.shape[1])

    colored_mask_img = Image.new('RGB', img.size, color=tuple(color))
    origin_img_rgba = img.convert('RGBA')

    mask_layer = Image.new('RGBA', img.size, (255,255,255,0)) # Start transparent
    mask_pil = Image.fromarray(mask_np * 255).convert('L')
    color_with_alpha = (*color, int(alpha * 255))
    colored_fill = Image.new('RGBA', img.size, color_with_alpha)
    mask_layer.paste(colored_fill, (0,0), mask_pil)
    
    origin_img_rgba.paste(mask_layer, (0,0), mask_layer) # Paste mask_layer onto itself (effectively blending)

    return origin_img_rgba

def denormalize(tens):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    return (tens*std)+mean

def make_coco_transforms(image_set, max_size, resize=False):
    normalize = TV.Compose([
        TV.ToTensor(),
        TV.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    scales = [480]
    if image_set == 'vae':
        return TV.Compose(
            [
                TV.RandomResize([480], max_size=832),
                normalize,
            ]
        )
        
    if image_set == 'train' or image_set == 'valid_u':
        return TV.Compose([
            TV.RandomSelect(
                TV.Compose([
                    TV.RandomResize(scales, max_size=max_size),
                    TV.Check(),
                ]),
                TV.Compose([
                    TV.RandomResize([400, 500, 600]),
                    TV.RandomSizeCrop(384, 600),
                    TV.RandomResize(scales, max_size=max_size),
                    TV.Check(),
                ])
            ),
            TV.RandomHorizontalFlip(),
            normalize,
        ])
        
    if image_set == 'valid' or image_set == 'val':
        return TV.Compose(
            [
                TV.RandomResize([480], max_size=832),
                normalize,
            ]
        )
    
    raise ValueError(f'unknown {image_set}')


class FrameSampler:
    @staticmethod
    def sample_local_frames(frame_id, vid_len, num_frames):
        sample_indx = []
        start_sample_id = max(frame_id - num_frames, 0)
        n_before_sample = min(frame_id, num_frames // 2)
        ids_before = random.sample(range(start_sample_id, frame_id), n_before_sample)

        end_sample_id = min(frame_id + num_frames, vid_len)
        n_after_sample = min(vid_len - frame_id - 1, num_frames // 2)
        ids_after = random.sample(range(frame_id, end_sample_id), n_after_sample)
        sample_indx.extend(ids_before)
        sample_indx.extend(ids_after)
        # if num_frames is odd, add frame_id
        if (len(sample_indx) < num_frames) and (frame_id not in sample_indx):
            sample_indx.append(frame_id)
        # if still not enough_frames, means we are close to the end
        # or start of the video; sample more
        if len(sample_indx) < num_frames:
            frame_pool = range(max(0, frame_id - num_frames*2), min(vid_len, frame_id + num_frames*2))
            done = FrameSampler.sample_from_pool(frame_pool, sample_indx, num_frames)
            if not done:
                while len(sample_indx) < num_frames:
                    samp_frame = random.sample(range(vid_len), 1)[0]
                    sample_indx.append(samp_frame)
                # raise Exception(f'[{frame_id}]:could not find {num_frames} sample in {vid_len} in pool {frame_pool}, having {sample_indx}')
                
        sample_indx.sort()
        return sample_indx

    @staticmethod
    def sample_from_pool(frame_pool, sample_indx, num_frames):
        iters = 0
        max_iter = len(frame_pool)*3
        while (len(sample_indx) < num_frames) and (iters < max_iter):
            samp_frame = random.sample(frame_pool, 1)[0]
            if samp_frame not in sample_indx:
                sample_indx.append(samp_frame)
            iters += 1
        
        return len(sample_indx) == num_frames

    @staticmethod
    def sample_global_frames(frame_id, vid_len, num_frames):
        # random sparse sample
        sample_indx = [frame_id]
        if num_frames != 1:
            # local sample
            sample_id_before = random.randint(1, 3)
            sample_id_after = random.randint(1, 3)
            local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >= global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    while len(sample_indx) < num_frames:
                        samp_frame = random.sample(range(vid_len), 1)[0]
                        sample_indx.append(samp_frame)

        sample_indx.sort()
        return sample_indx


class VideoEvalDataset(Dataset):
    def __init__(self, vid_folder, frames, ext='.jpg', target_h=480, target_w=832):
        super().__init__()
        self.vid_folder = vid_folder
        self.frames = frames
        self.vid_len = len(frames)
        self.ext = ext
        self.origin_w, self.origin_h = Image.open(join(vid_folder, frames[0]+ext)).size
        self.transform = TF.Compose([
            TF.Lambda(lambda img: resize_to_multiple_of_32_and_fit_box(img, multiple=16, target_h=target_h, target_w=target_w)),
            TF.ToTensor(),
            TF.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
            
    def __len__(self):
        return self.vid_len
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        img_path = join(self.vid_folder, frame + self.ext)
        img = Image.open(img_path).convert('RGB')
        
        return self.transform(img), idx

def check_shape(imgs):
    # b, c, t, h, w  vae and dit requirement.
    if imgs.shape[-1] % 16 == 0 and imgs.shape[-2] % 16 == 0:
        return imgs
    
    else:
        # Assuming you want to resize mask_input to video_input's size
        target_h = math.ceil(imgs.shape[-2] / 16) * 16
        target_w = math.ceil(imgs.shape[-1] / 16) * 16
        
        # Reshape mask_input to [B*F, C_mask, H_mask, W_mask] temporarily for interpolation
        original_shape = imgs.shape
        imgs_reshaped = imgs.view(-1, original_shape[1], original_shape[3], original_shape[4])
        
        imgs_input = F.interpolate(
            imgs_reshaped,
            size=(target_h, target_w),
            mode='bilinear', 
            #align_corners=False
        )
        
        # Reshape back to original 5D format [B, C_mask, F, H_video, W_video]
        imgs = imgs_input.view(original_shape[0], original_shape[1], original_shape[2], target_h, target_w)

    return imgs

