import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('WanPipeline R-VOS training and inference scripts.', add_help=False)

    # --- Training & Optimization Parameters ---
    parser.add_argument('--lr', default=5e-5, type=float, help="Main learning rate.")
    parser.add_argument('--lr_drop', default=[30], type=int, nargs='+',
                        help='Epochs to drop learning rate.')
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size for training.")
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="Weight decay.")
    parser.add_argument('--epochs', default=50, type=int, help="Total number of training epochs.")
    parser.add_argument('--clip_max_norm', default=1.0, type=float,
                        help='Gradient clipping max norm.')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='Enable Automatic Mixed Precision (AMP) training.')
    parser.add_argument('--resume', default='', help='Path to checkpoint to resume from.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch if resuming training.')
    parser.add_argument('--output_dir', default='output',
                        help='Path where to save checkpoints and logs. Empty for no saving.')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing (e.g., "cuda", "cpu").')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--cache_mode', default=False, action='store_true',
                        help='Whether to cache images/videos on memory.')
    parser.add_argument('--use_gradient_ckpt', default=True, help='enable_gradient_checkpointing for DiT')
    parser.add_argument('--use_dvi', default=False, help='Use DVI strategy')

    # --- Distributed Training Parameters ---
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes.')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training.')
    
    # --- Video & Frame Dimensions ---
    parser.add_argument('--num_frames', default=17, type=int,
                        help="Number of frames processed per video clip by the pipeline.")
    parser.add_argument('--image_size', default=512, type=int,
                        help="Input image/frame resolution (e.g., 512x512).")
    
    # --- Dataset Parameters ---
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name (e.g., "ytvos", "davis").')
    parser.add_argument('--ytvos_path', type=str, default='datasets/refer_youtube_vos',
                        help='Path to the YouTube-VOS dataset.')
    parser.add_argument('--davis_path', type=str, default='datasets/refer_davis',
                        help='Path to the DAVIS dataset.')
    parser.add_argument('--a2d_path', type=str, default='datasets/a2d_sentences',
                        help='Path to the A2D-Sentences dataset.')
    parser.add_argument('--jhmdb_path', type=str, default='datasets/jhmdb_sentences',
                        help='Path to the JHMDB-Sentences dataset.')
    parser.add_argument('--mevis_path', type=str, default='datasets/MeViS',
                        help='Path to the mevis dataset.')
    parser.add_argument('--coco_path', type=str, default='datasets/coco',
                        help='Path to the coco dataset.')
    parser.add_argument('--augm_resize', default=False, action='store_true',
                        help="Enable data augmentation with random resizing")
    parser.add_argument('--max_skip', default=3, type=int,
                        help="Maximum skip frame number when sampling frames.")
    parser.add_argument('--max_size', default=832, type=int,
                        help="Max longer side size for frame preprocessing.")
    parser.add_argument('--binary', action='store_true',
                        help='Use binary mask segmentation (e.g., for A2D/JHMDB).')
    
    # --- Evaluation / Inference Parameters ---
    parser.add_argument('--eval', default=False, action='store_true', help='Run evaluation only.')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Binary threshold for mask output during inference.')
    parser.add_argument('--ngpu', default=8, type=int,
                        help='Number of GPUs to use for inference (e.g., for parallel processing).')
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test', 'valid_u', 'train'],
                        help='Dataset split for evaluation (e.g., "valid", "test").')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to visualize the masks during inference.')

    return parser

