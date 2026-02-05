# main.py

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import sys # Import sys for stderr
import numpy as np
import torch
# Make sure DistributedSampler is imported if used
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset 
from engine import train_one_epoch, write_log 
from models.wan_rvos import build_dit as build_model
from models.text import TextProcessor
import opts
import warnings
from peft import LoraConfig, get_peft_model, TaskType
warnings.filterwarnings("ignore")
from diffusers.training_utils import EMAModel

def save_file(cfg_path, script_path):
    with open(script_path, 'r') as file: 
        content = file.read()
    with open(cfg_path + '/' + script_path.split('/')[-1], 'w') as file: 
        file.write(content) 
        print(f"File {script_path} saved.")
        
def load_modeles(device):
    # use default vae is ok
    model_id = "Wan2.1-T2V-1.3B-Diffusers" 
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return vae.eval(), tokenizer, text_encoder.eval()
   
def main(args):

    utils.init_distributed_mode(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_file(args.output_dir, 'engine.py')
    save_file(args.output_dir, 'models/wan_rvos.py')
    save_file(args.output_dir, 'models/transformer.py')
    save_file(args.output_dir, 'main.py')
    save_file(args.output_dir, 'datasets/transform_utils.py')
    save_file(args.output_dir, 'datasets/mevis.py')

    log_file_path = None
    if utils.is_main_process():
        log_file_path = os.path.join(args.output_dir, "log.txt")
        write_log(log_file_path, f"Output Directory: {args.output_dir}")
        write_log(log_file_path, "Command Line Arguments:")
        sorted_args = sorted(vars(args).items())
        for arg, value in sorted_args:
            write_log(log_file_path, f"  {arg}: {value}")
        write_log(log_file_path, "="*50)

        config_path = os.path.join(args.output_dir, "configs.txt")
        try:
            with open(config_path, 'w') as f:
                for arg, value in sorted_args:
                     f.write(f"{arg}: {value}\n")
            print(f"Saved configuration to {config_path}")
        except Exception as e:
             print(f"Error saving configs.txt: {e}", file=sys.stderr)
             write_log(log_file_path, f"Error saving configs.txt: {e}")

    print(f'\n **** Run on {args.dataset_file} dataset. **** \n')
    write_log(log_file_path, f'**** Run on {args.dataset_file} dataset. ****')
    
    if args.dataset_file == 'davis':
        args.save_period = 15
    if args.dataset_file == 'ytvos':
        args.save_period = 2
    if args.dataset_file == 'mevis':
        args.save_period = 1
    if args.dataset_file == 'pretrain':
        args.save_period = 1
        
        
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    write_log(log_file_path, f"Rank {utils.get_rank()} initialized with Seed: {seed} (Base: {args.seed})")

    vae, tokenizer, text_encoder = load_modeles(device)
    text_processor = TextProcessor(tokenizer, text_encoder)

    write_log(log_file_path, "Pre-trained components loaded.")
    write_log(log_file_path, "Building model")
    model = build_model(args)

    if not args.amp:
        model = model.to(torch.bfloat16)

    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,
        model_cls=type(model),
        model_config=model.config
    )
    ema_model.to(dtype=torch.float32)
    ema_model.shadow_params = [p.to(dtype=torch.float32) for p in ema_model.shadow_params]
    ema_model.shadow_params = [p.to(device=device) for p in ema_model.shadow_params]
    ema_model.to(device)

    write_log(log_file_path, "Model built and moved to device.")
    model_without_ddp = model
    
    if args.resume:
        print(f"[Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            print("[Info] Loading base model weights (to restore buffers)...")
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        
        if 'ema_model' in checkpoint:
            print("[Info] Found 'ema_model'. Applying EMA weights...")
            ema_helper = EMAModel(
                model_without_ddp.parameters(),
                decay=0.9999,
                model_cls=type(model_without_ddp),
                model_config=model_without_ddp.config
            )
            ema_helper.load_state_dict(checkpoint['ema_model'])
            ema_helper.copy_to(model_without_ddp.parameters())
            print("[Info] EMA weights applied successfully.")
            del ema_helper
            torch.cuda.empty_cache()
        else:
            print("[Warning] No EMA found in checkpoint, using standard weights.")
    else:
        raise ValueError('Please specify the checkpoint for inference using --resume.')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
        write_log(log_file_path, "Using DistributedDataParallel.")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', n_parameters)
    write_log(log_file_path, f'Number of trainable params: {n_parameters}')

    
    param_dicts = [
        {
            "params": [],
            "lr": args.lr, 
            "name": "base"
        },]
    for n, p in model.named_parameters():
        param_dicts[0]["params"].append(p)
        

    optimizer = bnb.optim.AdamW8bit(
        param_dicts, 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop, gamma=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print("\n **** Using AMP? {}. **** \n".format(args.amp))
    write_log(log_file_path, f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    write_log(log_file_path, f"LR Scheduler: MultiStepLR (milestones={args.lr_drop}, gamma=0.1)")
    write_log(log_file_path, f"Using AMP: {args.amp}")
    write_log(log_file_path, f"Parameter groups setup (L respectivos): "
              f"{[d['lr'] for d in optimizer.param_groups]}")


    # === Datasets ===
    dataset_train = None
    data_loader_train = None
    sampler_train = None 
    if not (args.eval and (args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb')):
        write_log(log_file_path, f"Building training dataset: {args.dataset_file}...")
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
        
        if args.combined_dataset is not None:
            dataset_train_combined = build_dataset(args.combined_dataset, image_set='train', args=args)
            dataset_train = ConcatDataset([dataset_train, dataset_train_combined])

        write_log(log_file_path, f"Training dataset size: {len(dataset_train)}")

        if args.distributed:
            if args.cache_mode and hasattr(samplers, 'NodeDistributedSampler'): # Check if NodeDistributedSampler exists
                sampler_train = samplers.NodeDistributedSampler(dataset_train)
                write_log(log_file_path, "Using NodeDistributedSampler for training.")
            else:
                sampler_train = DistributedSampler(dataset_train)
                write_log(log_file_path, "Using DistributedSampler for training.")
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            write_log(log_file_path, "Using RandomSampler for training.")

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                       pin_memory=True) 
        write_log(log_file_path, f"Training DataLoader created (Batch Size: {args.batch_size}, Num Workers: {args.num_workers}).")

    # === Training Loop ===
    print("\n **** Start training, total epochs: {}, starting from epoch: {}. **** \n".format(args.epochs, args.start_epoch))
    write_log(log_file_path, f"======== Starting Training Loop (Epochs {args.start_epoch} to {args.epochs-1}) ========")
    start_time = time.time()
    total_itr_num = 0 

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()

        if epoch > 0 and args.reload_dataset_per_epoch: 
            write_log(log_file_path, f"Reloading training dataset for Epoch {epoch}...")
            dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
            if args.distributed:
                if args.cache_mode and hasattr(samplers, 'NodeDistributedSampler'):
                    sampler_train = samplers.NodeDistributedSampler(dataset_train)
                else:
                    sampler_train = DistributedSampler(dataset_train)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                           collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                           pin_memory=True)
            print(f"Reloaded dataset for epoch {epoch}.")
            write_log(log_file_path, f"Dataset reloaded for Epoch {epoch}.")

        if args.distributed and sampler_train is not None:
             if hasattr(sampler_train, 'set_epoch'):
                 sampler_train.set_epoch(epoch)
             else:
                 print(f"Warning: Sampler {type(sampler_train)} does not have set_epoch method.", file=sys.stderr)

        
        # === Train One Epoch ===
        train_stats, current_total_itr = train_one_epoch(
            args, model, ema_model, vae, text_processor, data_loader_train, 
            optimizer, grad_scaler, device, epoch,
            args.clip_max_norm, total_itr_num, log_file_path)
        total_itr_num = current_total_itr 
        
        epoch_end_time = time.time()
        epoch_duration_str = str(datetime.timedelta(seconds=int(epoch_end_time - epoch_start_time)))
        print(f"\n **** Epoch {epoch} finished. Time cost: {epoch_duration_str}. **** \n")

        # === Step LR Scheduler ===
        lr_scheduler.step()
        write_log(log_file_path, f"LR Scheduler stepped after epoch {epoch}. New LR (Group 0): {optimizer.param_groups[0]['lr']:.6f}")

        # === Save Checkpoint ===
        if args.output_dir: 
            save_dict = {
                'model': model_without_ddp.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch, 
                'args': args,
            }
            if args.amp:
                 save_dict['grad_scaler'] = grad_scaler.state_dict()

            latest_path = output_dir / 'checkpoint_latest.pth'
            utils.save_on_master(save_dict, latest_path)

            if (epoch + 1) % args.save_period == 0 or epoch == args.epochs - 1:
                period_path = output_dir / f'checkpoint{epoch:04}.pth'
                utils.save_on_master(save_dict, period_path)
                write_log(log_file_path, f"Checkpoint saved: {period_path}") # Log periodic save

            # Minimal log for latest save to avoid clutter
            if utils.is_main_process() and (epoch + 1) % args.print_freq == 0: # Log less frequently
                  write_log(log_file_path, f"Checkpoint updated: {latest_path}")


        # === Log Epoch Summary Stats ===
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'duration_seconds': int(epoch_end_time - epoch_start_time),
                     'n_parameters': n_parameters}
        # Add LRs to summary
        for i, pg in enumerate(optimizer.param_groups):
            log_stats[f'lr_group{i}'] = pg['lr']

    # === End of Training ===
    total_time_secs = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time_secs)))
    print('\n **** Total training time: {} **** \n'.format(total_time_str))
    write_log(log_file_path, f"======== Training Finished. Total time: {total_time_str} ========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FlowRVS training and evaluation script', parents=[opts.get_args_parser()])
    parser.add_argument('--print_freq', default=100, type=int, help="Frequency of logging training status (iterations)")
    parser.add_argument('--save_period', default=20, type=int, help="Save checkpoint every N epochs")
    parser.add_argument('--eval_period', default=2, type=int, help="Evaluate on validation set every N epochs (if eval_during_train is True)")
    parser.add_argument('--eval_during_train', action='store_true', help="Perform evaluation on validation set during training")
    parser.add_argument('--reload_dataset_per_epoch', action='store_true', help="Reload the training dataset every epoch (usually not needed)")
    parser.add_argument('--find_unused_params', action='store_true', help="Set find_unused_parameters=True for DDP (needed for some models)")
    parser.add_argument('--save_eval_vis', default=True, action='store_true', help="Evaluate on validation set every N epochs (if eval_during_train is True)")
    parser.add_argument('--ft', default=False, action='store_true', help="2-stages finetuning")
    parser.add_argument('--ft_checkpoint', type=str, default=None, help="Finetuned ckpt.")
    parser.add_argument('--combined_dataset', type=str, default=None, help="Added dataset.")
    
    args = parser.parse_args()

    base_lr = args.lr
    base_lr_backbone = args.lr_backbone if hasattr(args, 'lr_backbone') else base_lr
    base_lr_text_encoder = args.lr_text_encoder if hasattr(args, 'lr_text_encoder') else base_lr
    base_lr_lora = args.lr_lora if hasattr(args, 'lr_lora') else base_lr 

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args) 
    print("Save results at: {}.".format(os.path.join(args.output_dir, "DVS_Annotations")))
