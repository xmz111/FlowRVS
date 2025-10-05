import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKLWan # Ensure you import Wan's specific VAE if it has a different class name/structure

class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)
    
    
class MaskVAEFinetuner(nn.Module):
    """
    A wrapper around Wan VAE for finetuning its decoder to output single-channel masks.
    """
    def __init__(self, args, vae_model_id, target_dtype=torch.bfloat16):
        super().__init__()
        print(f"Loading VAE from {vae_model_id} (subfolder 'vae')...")
        self.vae = AutoencoderKLWan.from_pretrained(vae_model_id, subfolder="vae", torch_dtype=target_dtype)
        print("VAE loaded.")

        original_conv_out = self.vae.decoder.conv_out
        in_channels = original_conv_out.in_channels
        out_dim = in_channels


        new_conv_out = WanCausalConv3d(
            in_channels=out_dim,
            out_channels=1,      
            kernel_size=3,
            padding=1
        )

        self.vae.decoder.conv_out = new_conv_out
        self.vae.decoder.conv_out.to(device=self.vae.device, dtype=target_dtype)

        for param in self.vae.encoder.parameters():
            param.requires_grad = False
        print("VAE Encoder frozen.")

        for param in self.vae.decoder.parameters():
            param.requires_grad = True
        print("VAE Decoder unfrozen.")
            
        self.vae.encoder.eval() 


    def forward(self, mask_input: torch.Tensor):
        """
        Forward pass for mask VAE finetuning.
        Args:
            mask_input (torch.Tensor): Input masks, expected to be (B, T, 1, H, W).
                                        Will be converted to (B*T, 3, H, W) for VAE encoder.
        Returns:
            reconstructed_mask_logits (torch.Tensor): Reconstructed mask logits, (B, T, 1, H, W).
        """

        mask_input_rgb = mask_input.repeat(1, 1, 3, 1, 1).transpose(1, 2) 
        mask_input_flat = mask_input_rgb * 2.0 - 1.0 

        with torch.no_grad():
            latent_dist = self.vae.encode(mask_input_flat.to(self.vae.dtype)).latent_dist
        mask_latent = latent_dist.sample() 
        reconstructed_mask_logits = self.vae.decode(mask_latent, return_dict=False)[0]

        return reconstructed_mask_logits

    def get_trainable_parameters(self):
        """
        Returns parameters that require gradients, specifically for finetuning.
        """
        return [p for p in self.parameters() if p.requires_grad]

