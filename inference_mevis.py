import os
import sys
import argparse
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
import json
from pathlib import Path
import opts
import warnings
warnings.filterwarnings("ignore")
import util.misc as utils
from models.wan_rvos import build_dit 
from models.text import TextProcessor 
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
from utils_inf import colormap # Assuming this utility exists for visualization
from transformers import AutoTokenizer # Needed for TextProcessor
from models.mask_vae_finetuner import MaskVAEFinetuner
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_dataset
from metrics import db_eval_boundary, db_eval_iou
from os.path import join
from datasets.transform_utils import VideoEvalDataset, vis_add_mask, check_shape, check_shape_big, vis_add_mask_new
from pycocotools import mask as cocomask

# colormap for visualization
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

def load_modeles(device):
    model_id = "Wan2.1-T2V-1.3B-Diffusers" # Or your local path
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(model_id, subfolder="scheduler") 
    return tokenizer, text_encoder, scheduler

def main(args):
    args.masks = True
    args.batch_size == 1
    args.eval = True

    print(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    print('Start inference')

    model, vae, text_encoder, scheduler = prepare()

    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, 'Annotations')
    os.makedirs(save_path_prefix, exist_ok=True)
    args.log_file = join(args.output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(sys.argv)+'\n')
        fp.writelines(str(args.__dict__)+'\n\n')        

    split = args.split
    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        os.makedirs(save_visualize_path_prefix, exist_ok=True)
        
    model = model.to(dtype=torch.bfloat16).eval()
    eval_mevis(args, model, vae, text_encoder, scheduler,\
        save_path_prefix, save_visualize_path_prefix, split=split)


    end_time = time.time()
    total_time = end_time - start_time

    print("Total inference time: %.4f s" %(total_time))
    
def prepare():

    device = torch.device(args.device)
    # 1. Load DiT Model (Main Model)
    model = build_dit(args)
    #model.to(device)

    # 2. Load VAE and Text Encoder (Auxiliary Models)
    tokenizer, text_encoder, scheduler = load_modeles(device)
    text_processor = TextProcessor(tokenizer, text_encoder)

    model_id = "Wan2.1-T2V-1.3B-Diffusers"   
    mask_vae = MaskVAEFinetuner(vae_model_id=model_id, target_dtype=torch.bfloat16)
    
    print(f"[Loading checkpoint from {args.vae_ckpt}]")
    vae_checkpoint = torch.load(args.vae_ckpt, map_location='cpu', weights_only=False)
    vae_state_dict = vae_checkpoint.get('model', vae_checkpoint)
    torch.save(vae_state_dict, 'decoder.pth')
    missing_keys, unexpected_keys = mask_vae.load_state_dict(vae_state_dict, strict=True)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print(f'Missing Keys: {missing_keys}')
    if len(unexpected_keys) > 0:
        print(f'Unexpected Keys: {unexpected_keys}')
    del vae_checkpoint

    # now vae is same as WAN VAE, but use tuned weight
    vae = mask_vae.vae.to(device).eval() 

    model_without_ddp = model 
    if args.dit_ckpt:
        print(f"[Loading checkpoint from {args.dit_ckpt}]")
        checkpoint = torch.load(args.dit_ckpt, map_location='cpu', weights_only=False)
        model_state_dict = checkpoint.get('model', checkpoint) 
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(model_state_dict, strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print(f'Missing Keys: {missing_keys}')
        if len(unexpected_keys) > 0:
            print(f'Unexpected Keys: {unexpected_keys}')
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference using --dit_ckpt.')
    
    return model_without_ddp, vae, text_processor, scheduler


def eval_mevis(args, model, vae, text_processor, scheduler, save_path_prefix, save_visualize_path_prefix, split='valid_u'):
    root = Path(args.mevis_path)
    img_folder = join(root, split, "JPEGImages")
    meta_file = join(root, split, "meta_expressions.json")
    gt_file = join(root, split, 'mask_dict.json')
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    if args.split == 'valid_u' or args.split == 'train':
        with open(gt_file, "r") as f:
            gt_data = json.load(f)        
    else:
        gt_data = None

    video_list = list(data.keys())
    progress = tqdm(
        total=len(video_list),
        ncols=0
    )
    f_log_vid = join(args.output_dir, 'log_metrics_byvid.txt')

    f_log = join(args.output_dir, 'log_metrics.txt')
    model.eval()
    out_dict = {}
    
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    mean_tensor = latents_mean.to(device=vae.device, dtype=vae.dtype)
    std_tensor = latents_std.to(device=vae.device, dtype=vae.dtype)
    
    # 1. For each video
    for i_v, video in enumerate(video_list):
        metas = [] # list[dict], length is number of expressions
        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        out_dict_per_vid = {}
        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas
    
        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            original_frames = meta[i]["frames"]

            video_len = len(original_frames) 

            frames = original_frames
            target_h, target_w = args.reso_h, args.reso_w
            vd = VideoEvalDataset(join(img_folder, video_name), frames, target_h=target_h, target_w=target_w)
            
            origin_w, origin_h = vd.origin_w, vd.origin_h
            dl = DataLoader(vd, batch_size=len(frames),
                    num_workers=args.num_workers, shuffle=False)
            origin_w, origin_h = vd.origin_w, vd.origin_h
            
            all_pred_masks = []
            # 3. for each clip
            for imgs, clip_frames_ids in dl:
                clip_frames_ids = clip_frames_ids.tolist()
                imgs = imgs.to(args.device)  # [eval_clip_window, 3, h, w]
                img_h, img_w = imgs.shape[-2:]
                size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                target = {"size": size, 'frame_ids': clip_frames_ids}
                t = imgs.shape[0]
                if (t - 1) % 4 != 0:
                    num_padding_frames = (4 - (t - 1) % 4) % 4
                    last_frame = imgs[-1:] 
                    padding_frames = last_frame.repeat(num_padding_frames, 1, 1, 1)
                    imgs = torch.cat([imgs, padding_frames], dim=0)
                    
                imgs = imgs.unsqueeze(0)
                with torch.no_grad():
                    imgs = check_shape(imgs)
                    x0_video_latent = vae.encode(imgs.transpose(1, 2).to(vae.dtype)).latent_dist.mean 
                    prompt_embeds, negative_prompt_embeds = text_processor.encode_prompt_and_cfg(
                        prompt=[exp], 
                        device=args.device,
                        dtype=vae.dtype,
                    )
                    
                    shift = 3
                    t_steps = torch.linspace(1.0, 0.001, args.num_steps + 1, dtype=torch.float).to(args.device)
                    timesteps = shift * t_steps / (1 + (shift - 1) * t_steps) * 1000
                    timesteps = timesteps[:-1]
                    timesteps = timesteps.round()

                    scheduler.set_timesteps(
                        num_inference_steps=len(timesteps), 
                        timesteps=timesteps.tolist(),
                        device=args.device
                    )
                    
                    x0_video_latent = (x0_video_latent - mean_tensor) / std_tensor
                    latents = x0_video_latent
                    for k, t in enumerate(timesteps):
                        timestep = t.expand(latents.shape[0])
                        noise_pred = model(
                            hidden_states=latents.to(model.dtype),
                            video_condition=x0_video_latent.to(model.dtype),
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=None, 
                            return_dict=True,
                        )[0]
                            
                        
                        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                        
                    latents = latents * std_tensor + mean_tensor
                    decoded_pixel_output = vae.decode(latents.detach())[0] 
                    decoded_pixel_output = F.interpolate(decoded_pixel_output.view(-1, 1, decoded_pixel_output.shape[-2], decoded_pixel_output.shape[-1]), 
                                size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
                    reconstructed_mask_probs = torch.sigmoid(decoded_pixel_output)
                    
                    reconstructed_mask_binary = (reconstructed_mask_probs > 0.5).float().squeeze(0).squeeze(1)
                    all_pred_masks.append(reconstructed_mask_binary.cpu())

            all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy()[:video_len]

            if args.split == 'valid_u' or args.split == 'train':
                h, w = all_pred_masks.shape[-2:]
                gt_masks = np.zeros((video_len, h, w), dtype=np.uint8)
                anno_ids = data[video]['expressions'][exp_id]['anno_id']
                for frame_idx, frame_name in enumerate(data[video]['frames']):
                    for anno_id in anno_ids:
                        mask_rle = gt_data[str(anno_id)][frame_idx]
                        if mask_rle:
                            gt_masks[frame_idx] += cocomask.decode(mask_rle)
                
                j = db_eval_iou(gt_masks, all_pred_masks).mean()
                f = db_eval_boundary(gt_masks, all_pred_masks).mean()
                out_dict[exp] = [j, f]
                out_dict_per_vid[exp] = [j, f]
                
                '''
                save_path = join(save_path_prefix, video_name, exp_id)
                os.makedirs(save_path, exist_ok=True)
                for j in range(video_len):
                    frame_name = original_frames[j]
                    mask = all_pred_masks[j].astype(np.float32) 
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frame_name + ".png")
                    mask.save(save_file)
                '''
            else:
                # save binary image
                save_path = join(save_path_prefix, video_name, exp_id)
                os.makedirs(save_path, exist_ok=True)
                for j in range(video_len):
                    frame_name = original_frames[j]
                    mask = all_pred_masks[j].astype(np.float32) 
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frame_name + ".png")
                    mask.save(save_file)
                
            if args.visualize:
                for t, frame in enumerate(original_frames):
                    # original
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path).convert('RGBA') # PIL image

                    # draw mask
                    source_img = vis_add_mask_new(source_img, all_pred_masks[t], color_list[i%len(color_list)])

                    # save
                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                    os.makedirs(save_visualize_path_dir, exist_ok=True)
                    save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                    source_img.save(save_visualize_path)
                    
        if args.split == 'valid_u' or args.split == 'train':
            J_score, F_score, JF = get_current_metrics(out_dict)
            J_score_vid, F_score_vid, JF_vid = get_current_metrics(out_dict_per_vid)
            out_str = f'{i_v}/{len(video_list)} J&F: {JF}\tJ: {J_score}\tF: {F_score}'
            out_str_vid = f'{i_v}/{len(video_list)} CURRENT J&F: {JF_vid}  J: {J_score_vid}  F: {F_score_vid}'
            if utils.is_main_process():
                with open(f_log, 'a') as fp:
                    fp.write(out_str + '\n')
                    fp.write(out_str_vid + '\n')
                with open(f_log_vid, 'a') as fp:
                    fp.writelines(out_str_vid + '\n')
            print('\n' + out_str + '\n' + out_str_vid)
        progress.update(1)

    if args.split == 'valid_u' or args.split == 'train':
        J_score, F_score, JF = get_current_metrics(out_dict)
        print(f'J: {J_score}')
        print(f'F: {F_score}')
        print(f'J&F: {JF}')
        
        return [J_score, F_score, JF]
    return [0, 0, 0]

def get_current_metrics(out_dict):
    j = [out_dict[x][0] for x in out_dict]
    f = [out_dict[x][1] for x in out_dict]

    J_score = np.mean(j)
    F_score = np.mean(f)
    JF = (np.mean(j) + np.mean(f)) / 2
    return J_score, F_score, JF
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser('FlowRVS inference script', parents=[opts.get_args_parser()])
    parser.add_argument('--print_freq', default=1, type=int, help="Frequency of logging training status (iterations)")
    parser.add_argument('--save_eval_vis', default=True, action='store_true', help="Evaluate on validation set every N epochs (if eval_during_train is True)")
    parser.add_argument('--num_steps', default=1, type=int, help="Evaluate on validation set every N epochs (if eval_during_train is True)")
    parser.add_argument('--dit_ckpt', default=None, type=str, help="DiT checkpoint")
    parser.add_argument('--vae_ckpt', default=None, type=str, help="VAE checkpoint for tuned decoder")
    parser.add_argument('--reso_h', default=480, type=int, help="VAE checkpoint for tuned decoder")
    parser.add_argument('--reso_w', default=832, type=int, help="VAE checkpoint for tuned decoder")
    args = parser.parse_args()
    main(args)
    print("Save results at: {}.".format(os.path.join(args.output_dir, "DVS_Annotations")))
