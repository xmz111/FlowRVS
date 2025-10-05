import os
import sys
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

# 从您的项目中导入必要的模块
import opts
import util.misc as utils
from models.wan_rvos import build_dit
from models.wan_rvos_2_2 import build_dit_big
from models.text import TextProcessor
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
from models.mask_vae_finetuner import MaskVAEFinetuner
from datasets.transform_utils import VideoEvalDataset, vis_add_mask_new, check_shape, check_shape_big
from utils_inf import colormap
from torch.utils.data import DataLoader
from moviepy import ImageSequenceClip

# 视觉化用的颜色列表
color_list = colormap().astype('uint8').tolist()


def extract_frames_from_mp4(video_path, output_folder):

    needs_extraction = True
    if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
        needs_extraction = False
        print(f'{output_folder} exists')

    if needs_extraction:
        os.makedirs(output_folder, exist_ok=True) 

        extract_cmd = f"ffmpeg -i \"{video_path}\" -loglevel error -vf fps={args.fps} \"{output_folder}/frame_%05d.png\""
        ret = os.system(extract_cmd)
        if ret != 0:
            if len(os.listdir(output_folder)) == 0:
                os.rmdir(output_folder)
            sys.exit(ret)
            
    frames_list = sorted([os.path.splitext(f)[0] for f in os.listdir(output_folder) if f.endswith('.png')])
    return output_folder, frames_list, '.png'


def prepare_models(args):
    device = torch.device(args.device)
    
    # 1. 加载主模型 DiT
    if args.big:
        model, _, _ = build_dit_big(args)
    else:
        model, _, _ = build_dit(args)

    # 2. 加载 VAE 和 Text Encoder
    model_id = "Wan2.2-TI2V-5B-Diffusers" if args.big else "Wan2.1-T2V-1.3B-Diffusers_download"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(model_id, subfolder="scheduler")
    
    text_processor = TextProcessor(tokenizer, text_encoder.to(device).eval())
    
    # 3. 加载修改过的 VAE Decoder (Mask Head)
    model_id = "Wan2.1-T2V-1.3B-Diffusers_download" # Or your local path
    if args.big:
        model_id = "Wan2.2-TI2V-5B-Diffusers"
        
    mask_head = MaskVAEFinetuner(
        args=args,
        vae_model_id=model_id,
        freeze_encoder=True,
        modify_type='add_output_head',
        target_dtype=vae.dtype, # Or based on your args setup
        
    )
    if args.big:
        ckpt_path = 'decoder_8_32/checkpoint0016.pth'
    else:
        ckpt_path = 'decoder_8_13/checkpoint0002.pth'
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        missing_keys, unexpected_keys = mask_head.load_state_dict(checkpoint['model'], strict=False)
        
    loaded_mask_output_head = None    
    vae.decoder = mask_head.vae.decoder
    
    vae.to(device).eval() 
    
    if args.resume:
        print(f"[Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_state_dict = checkpoint.get('model', checkpoint) 

        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print(f'Missing Keys: {missing_keys}')
        if len(unexpected_keys) > 0:
            print(f'Unexpected Keys: {unexpected_keys}')
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference using --resume.')
    
    model.to(device).to(torch.bfloat16).eval() # 使用 bfloat16 加速并节约显存
    vae.to(torch.bfloat16)
    text_processor.text_encoder.to(torch.bfloat16)

    return model, vae, text_processor, scheduler


def inference_single_video(args, model, vae, text_processor, scheduler, video_path, text_prompts):
    fname, ext = os.path.splitext(os.path.basename(video_path))
    if ext.lower() == '.mp4':
        temp_frames_folder = os.path.join(args.output_dir, f"frames_{fname}")
        frames_folder, frames_list, frame_ext = extract_frames_from_mp4(video_path, temp_frames_folder)
    elif os.path.isdir(video_path):
        frames_folder = video_path
        all_files = os.listdir(frames_folder)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            frames_list = []
            frame_ext = None
        else:
            frames_list = sorted([os.path.splitext(f)[0] for f in image_files])
            frame_ext = os.path.splitext(image_files[0])[1]
    else:
        raise ValueError(f"不支持的输入路径格式: {video_path}")


    mask_mean = [0.102, 1.785, -0.280, 1.776, -0.626, 0.370, 0.763, 0.196, 1.262, 2.408, -0.378, -1.655, 0.211, 0.138, 1.555, -0.497, -0.782, -0.336, -0.763, 0.303, 0.768, -2.598, -1.233, -0.372, 0.809, 0.269, -0.292, 0.048, -0.483, 0.536, -1.342, 0.157, -0.217, -0.917, 0.931, 0.596, 0.671, -2.088, 0.385, 3.371, -0.127, -1.399, 0.348, -0.107, -0.390, -0.299, -0.583, 0.964] if args.big else [-2.218, -0.402, -3.687, -3.968, 1.125, 0.099, 2.156, 1.351, 5.062, 1.968, -2.921, -2.140, 0.671, 0.859, -3.921, -3.015]
    mask_std = [0.227, 1.211, 0.524, 1.102, 0.511, 0.291, 0.366, 0.266, 0.745, 1.225, 0.451, 1.184, 0.341, 0.524, 1.087, 0.506, 0.601, 0.325, 0.529, 0.486, 0.443, 1.606, 1.147, 0.396, 0.685, 0.300, 0.228, 0.503, 0.376, 0.344, 0.818, 0.341, 0.174, 0.439, 0.619, 0.452, 0.413, 1.564, 0.526, 2.025, 0.280, 1.190, 0.349, 0.253, 0.379, 0.273, 0.669, 0.636] if args.big else [1.976, 1.046, 2.234, 3.015, 0.839, 1.046, 1.585, 1.101, 3.515, 1.664, 2.281, 1.507, 1.132, 0.878, 3.484, 2.234]
    
    device, dtype = vae.device, vae.dtype
    mean_tensor = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std_tensor = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    mean_tensor_mask = torch.tensor(mask_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std_tensor_mask = torch.tensor(mask_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)

    target_h, target_w = (480, 832)
    vd = VideoEvalDataset(frames_folder, frames_list, frame_ext, target_h=target_h, target_w=target_w)
    dl = DataLoader(vd, batch_size=len(frames_list), num_workers=args.num_workers, shuffle=False)
    origin_w, origin_h = vd.origin_w, vd.origin_h

    (imgs, _) = next(iter(dl))
    imgs = imgs.to(device)

    t = imgs.shape[0]
    original_len = t
    if (t - 1) % 4 != 0:
        num_padding_frames = (4 - (t - 1) % 4) % 4
        padding_frames = imgs[-1:].repeat(num_padding_frames, 1, 1, 1)
        imgs = torch.cat([imgs, padding_frames], dim=0)
    
    imgs = imgs.unsqueeze(0)
    
    with torch.no_grad():
        imgs = check_shape_big(imgs) if args.big else check_shape(imgs)
        x0_video_latent = vae.encode(imgs.transpose(1, 2).to(vae.dtype)).latent_dist.mean
        x0_video_latent = (x0_video_latent - mean_tensor) / std_tensor

        for i, prompt in enumerate(tqdm(text_prompts, desc=f"Processing prompts for {fname}")):
            prompt_embeds, _ = text_processor.encode_prompt_and_cfg(
                prompt=[prompt], device=device, dtype=dtype, do_classifier_free_guidance=args.cfg
            )

            shift = 3
            t_steps = torch.linspace(1.0, 0.001, args.num_steps + 1, device=device)
            timesteps = shift * t_steps / (1 + (shift - 1) * t_steps) * 1000
            scheduler.set_timesteps(num_inference_steps=args.num_steps, device=device)
            timesteps = scheduler.timesteps

            latents = x0_video_latent.clone()
            for t in tqdm(timesteps, leave=False, desc="Diffusion steps"):
                timestep = t.expand(latents.shape[0])
                noise_pred = model(
                    hidden_states=latents.to(model.dtype),
                    video_condition=x0_video_latent.to(model.dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                )[0]
                latents = scheduler.step(noise_pred, t, latents)[0]

            latents = latents * std_tensor_mask + mean_tensor_mask
            decoded_pixel_output = vae.decode(latents.detach())[0]
            decoded_pixel_output = F.interpolate(decoded_pixel_output.view(-1, 1, decoded_pixel_output.shape[-2], decoded_pixel_output.shape[-1]),
                                size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
            reconstructed_mask_probs = torch.sigmoid(decoded_pixel_output)
            all_pred_masks = (reconstructed_mask_probs > 0.5).float().cpu().squeeze(0).squeeze(1)
            all_pred_masks = all_pred_masks[:original_len].numpy() 

            
            prompt_sanitized = "".join(c if c.isalnum() else "_" for c in prompt)
            
            clip_source_list = []
            
            if args.save_fig:
                save_visualize_path_dir = os.path.join(args.output_dir, fname, prompt_sanitized)
                os.makedirs(save_visualize_path_dir, exist_ok=True)
                print(f"Saving frame visualizations to: {save_visualize_path_dir}")

            for frame_idx, frame_name in enumerate(frames_list):
                img_path = os.path.join(frames_folder, frame_name + frame_ext)
                source_img = Image.open(img_path).convert('RGBA')
                source_img = vis_add_mask_new(source_img, all_pred_masks[frame_idx], color_list[i % len(color_list)])
                
                if args.save_fig:
                    save_path = os.path.join(save_visualize_path_dir, f"{frame_name}.png")
                    source_img.save(save_path)
                    clip_source_list.append(save_path)
                else:
                    frame_as_array = np.array(source_img)
                    clip_source_list.append(frame_as_array)

            if clip_source_list:
                video_output_path = os.path.join(args.output_dir, f"{prompt_sanitized}_output.mp4")
                os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

                fps = args.fps 
                clip = ImageSequenceClip(clip_source_list, fps=fps)
                clip.write_videofile(video_output_path, codec='libx264', logger=None)

                print(f"Video saved to: {video_output_path}")


def main(args):
    utils.init_distributed_mode(args)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    model, vae, text_processor, scheduler = prepare_models(args)

    inference_single_video(args, model, vae, text_processor, scheduler, args.input_path, args.text_prompts)

    total_time = time.time() - start_time
    print(f"Time Consuming: {total_time:.4f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference Script', parents=[opts.get_args_parser()])
    parser.add_argument('--input_path', type=str, required=True, help='video .mp4 path')
    parser.add_argument('--text_prompts', type=str, required=True, nargs='+', help='text')
    parser.add_argument('--num_steps', default=4, type=int, help='Inference steps')
    parser.add_argument('--fps', default=24, type=int, help='Video FPS')
    parser.add_argument('--cfg', action='store_true', help='cfg')
    parser.add_argument('--save_fig', default=False, action='store_true',
                        help='Save figures')
    args = parser.parse_args()
    main(args)