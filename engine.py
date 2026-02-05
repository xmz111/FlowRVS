import math
import time
import torch.nn.functional as F
import os
import sys
from typing import Iterable
import torch
import torch.distributed as dist
import util.misc as utils 
import datetime 
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch.nn as nn
import math
from sample import boundary_biased_sample
from datasets.transform_utils import check_shape

def write_log(log_file_path, message):
    """Appends a timestamped message to the specified log file if on main process."""
    if not utils.is_main_process() or log_file_path is None:
        return
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} - {message}"
    print(message)
    try:
        with open(log_file_path, 'a') as log_f:
            log_f.write(log_entry + '\n')
    except Exception as e:
        print(f"Rank {utils.get_rank()} Error writing to log file {log_file_path}: {e}", file=sys.stderr)


def train_one_epoch(args, model: torch.nn.Module, ema_model: torch.nn.Module, vae: torch.nn.Module, 
                        text_processor: None, 
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        grad_scaler: torch.cuda.amp.GradScaler,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        total_itr_num=0, log_file_path=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    for i, group in enumerate(optimizer.param_groups):
        group_name = group.get('name', f'group_{i}')
        metric_logger.add_meter(f'lr_{group_name}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    write_log(log_file_path, f"Starting Epoch: [{epoch}]")

    itr_counter = 0
    max_iter = len(data_loader)
    end = time.time()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    mean_tensor = latents_mean.to(device=device, dtype=vae.dtype)
    std_tensor = latents_std.to(device=device, dtype=vae.dtype)

    for samples, targets in data_loader:
        optimizer.zero_grad()
        data_time = time.time() - end
        # --- Data preparation for MyWanPipeline ---
        # `samples` directly corresponds to `video_input` (B, T, C, H, W)
        # video_input = samples.to(device) # Original video frames, pixel-level
        video_input = samples.tensors.transpose(1, 2).to(device)
        video_input = check_shape(video_input)
        text_input = [t["caption"] for t in targets]

        mask_list_from_dataset: List[torch.Tensor] = []
        for t_item in targets:
            # t_item["masks"] is of shape [T, H, W]
            mask_tensor = t_item["masks"].float()
            # Add channel dimension: [T, H, W] -> [T, 1, H, W]
            mask_tensor = mask_tensor.unsqueeze(1) 
            mask_list_from_dataset.append(mask_tensor)
        
        # Batch the masks: List[T, 1, H, W] -> B, T, 1, H, W
        mask_input = torch.stack(mask_list_from_dataset, dim=0).transpose(1, 2).to(device)
        if mask_input.shape[-2:] != video_input.shape[-2:]:
            target_h = video_input.shape[-2]
            target_w = video_input.shape[-1]

            original_shape = mask_input.shape
            mask_input_reshaped = mask_input.view(-1, original_shape[1], original_shape[3], original_shape[4])
            
            resized_mask_input = F.interpolate(
                mask_input_reshaped,
                size=(target_h, target_w),
                mode='nearest', 
            )
            
            # Reshape back to original 5D format [B, C_mask, F, H_video, W_video]
            mask_input = resized_mask_input.view(original_shape[0], original_shape[1], original_shape[2], target_h, target_w)

        
        WAN_PRETRAINED_TOTAL_TIMESTEPS = 1000
        
        prompt_embeds, negative_prompt_embeds = text_processor.encode_prompt_and_cfg(
                prompt=text_input, 
                device=device,
                dtype=vae.dtype,
            )
        x0 = vae.encode(video_input.to(vae.dtype)).latent_dist.sample()
        mask_input_flat = mask_input.repeat(1, 3, 1, 1, 1).to(vae.dtype) * 2.0 - 1.0 
        x1 = vae.encode(mask_input_flat).latent_dist.mean

        x0 = (x0 - mean_tensor) / std_tensor

        timestep_for_dit, t_expanded = boundary_biased_sample(
            total_train_timesteps=WAN_PRETRAINED_TOTAL_TIMESTEPS,
            num_samples_per_batch=args.batch_size,
            sampling_strategy="bias_video", 
            bias_param=0.0,                
            special_timesteps=[1000],         
            special_freq=0.5,              
            device=device
        )

        u_t = x0 - x1 
        x_t_input =  t_expanded * x0 + (1.0 - t_expanded) * x1
    
        # --- Model Forward Pass ---
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
            if not args.amp:
                x_t_input = x_t_input.to(vae.dtype)
            predicted_output = model(
                hidden_states=x_t_input,
                video_condition=x0, 
                timestep=timestep_for_dit,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=None, 
                return_dict=True,
            )[0]

            predicted_v_t = predicted_output 
            losses = F.mse_loss(predicted_v_t, u_t)
            
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if args.amp:
            grad_scaler.scale(losses).backward() 
            grad_scaler.unscale_(optimizer)
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    (p for group in optimizer.param_groups for p in group['params']),
                    max_norm, error_if_nonfinite=False
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    (p for group in optimizer.param_groups for p in group['params']),
                    max_norm, error_if_nonfinite=False
                )
            optimizer.step()

        actual_model = model.module if hasattr(model, 'module') else model
        ema_model.step(actual_model.parameters())
  
        if not math.isfinite(loss_value):
            err_msg = "\n **** Loss is {}, stopping training. **** \n".format(loss_value)
            print(err_msg, file=sys.stderr)
            sys.exit(1)


        metric_logger.update(loss=loss_value)
        metric_logger.update(grad_norm=grad_total_norm.item())

        batch_time = time.time() - end
        end = time.time()

        if torch.isfinite(grad_total_norm):
            metric_logger.update(grad_norm=grad_total_norm.item())
        
        metric_logger.update(time=batch_time, data=data_time)

        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            metric_logger.update(**{f'lr_{group_name}': group['lr']})
        
        if log_file_path and itr_counter % args.print_freq == 0:
            eta_seconds = metric_logger.time.global_avg * (max_iter - itr_counter)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            mem_mb_val = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 if torch.cuda.is_available() else 0

            log_str_parts = [
                f"Epoch: {epoch}",
                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"[{itr_counter}/{max_iter}]",
                f"eta: {eta_string}",
                f"{str(metric_logger)}", 
                f"max mem: {mem_mb_val:.0f}M" 
            ]    
            iter_message = " | ".join(log_str_parts)
            write_log(log_file_path, iter_message)

        itr_counter += 1

    metric_logger.synchronize_between_processes()

    avg_stats_str = "Averaged stats: " + str(metric_logger)
    print(avg_stats_str)
    write_log(log_file_path, f"End of Epoch: [{epoch}] {avg_stats_str}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total_itr_num + itr_counter
