import torch
import torch.nn.functional as F
from torch import nn
import os
import copy
import warnings
warnings.filterwarnings("ignore")
from models.transformer import WanTransformer3DModel
from safetensors.torch import load_file as load_safetensors

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

def adapt_dit_for_concat_input(transformer: WanTransformer3DModel, video_latent_channels: int, mask_latent_channels: int):
    """
    Adapts the transformer's patch embedding and proj_out layers
    to accept the new combined video latent + mask latent input channels.
    It copies weights for original video channels and initializes new channels (for mask) with zeros.
    """
    new_in_channels = video_latent_channels + mask_latent_channels
    
    old_patch_embedding = transformer.patch_embedding
    old_in_channels = old_patch_embedding.in_channels
    old_out_channels = old_patch_embedding.out_channels
    old_kernel_size = old_patch_embedding.kernel_size
    old_stride = old_patch_embedding.stride
    old_padding = old_patch_embedding.padding
    old_bias = old_patch_embedding.bias is not None

    if old_in_channels == new_in_channels:
        print("Transformer's input channels already match the R-VOS requirement. No input adaptation needed.")

    if old_in_channels != video_latent_channels:
        print(f"WARNING: Original transformer in_channels ({old_in_channels}) does not match expected video latent channels ({video_latent_channels}). "
              "Weight copying might be incorrect. Please verify your `video_latent_channels` matches the pre-trained transformer's input.")

    new_patch_embedding = nn.Conv3d(
        new_in_channels,
        old_out_channels,
        kernel_size=old_kernel_size,
        stride=old_stride,
        padding=old_padding,
        bias=old_bias
    ).to(old_patch_embedding.weight.device, old_patch_embedding.weight.dtype)

    with torch.no_grad():
        new_patch_embedding.weight[:, :old_in_channels, :, :, :].copy_(old_patch_embedding.weight)
        new_patch_embedding.weight[:, old_in_channels:, :, :, :].zero_()
        if old_bias:
            if new_patch_embedding.bias.shape[0] < old_patch_embedding.bias.shape[0]:
                raise ValueError("New patch_embedding bias is smaller than old_patch_embedding bias.")
            new_patch_embedding.bias.copy_(old_patch_embedding.bias)

    transformer.patch_embedding = new_patch_embedding
    print(f"Transformer's patch embedding adapted from {old_in_channels} to {new_in_channels} input channels.")


def build_dit(args):
    device = torch.device(args.device)

    model_id = "Wan2.1-T2V-1.3B-Diffusers"
    config = WanTransformer3DModel.load_config(model_id, subfolder="transformer")
    config["use_dvi"] = args.use_dvi  # 假设 args 里有这个布尔值

    transformer = WanTransformer3DModel(**config)
    
    weights_path_part1 = os.path.join(model_id, "transformer", "diffusion_pytorch_model-00001-of-00002.safetensors")
    weights_path_part2 = os.path.join(model_id, "transformer", "diffusion_pytorch_model-00002-of-00002.safetensors")

    loaded_state_dict_part1 = load_safetensors(weights_path_part1)
    loaded_state_dict_part2 = load_safetensors(weights_path_part2)

    transformer.load_state_dict(loaded_state_dict_part1, strict=False)
    transformer.load_state_dict(loaded_state_dict_part2, strict=False)
    
    if args.use_gradient_ckpt:
        transformer.enable_gradient_checkpointing()
    if args.use_dvi:
        video_latent_channels, mask_latent_channels = 16, 16
        adapt_dit_for_concat_input(transformer, video_latent_channels, mask_latent_channels)
    transformer.to(device)
    return transformer
