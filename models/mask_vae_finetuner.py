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
    
'''
class MaskOutputHead(nn.Module):
    """
    一个小型的头部网络，用于将Wan VAE解码器输出的3通道RGB-like特征，
    转换为单通道的掩码 logits。
    """
    def __init__(self, vae_output_channels=3):
        super().__init__()
        # 最简单的实现：一个1x1卷积，将通道从3缩减到1
        # 您可以根据需要添加更多层，例如一个激活函数，或者一个小型的块
        self.conv_1x1 = nn.Conv2d(in_channels=vae_output_channels, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # 注意：这里不使用Sigmoid，因为我们将返回 logits，然后与BCEWithLogitsLoss一起使用

    def forward(self, x):
        # x 预期形状为 (B*T, 3, H, W)，来自 VAE 解码器的输出
        B_original, C, T_output, H_out, W_out = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B_original * T_output, C, H_out, W_out)
        x = self.conv_1x1(x).view(B_original, T_output, 1, H_out, W_out).transpose(1, 2)
        # x = self.sigmoid(x)
        return x # 输出形状为 (B*T, 1, H, W)
'''

class MaskOutputHead(nn.Module):
    """
    一个小型的头部网络，用于将Wan VAE解码器输出的3通道RGB-like特征，
    转换为单通道的掩码 logits。
    """
    def __init__(self, vae_output_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(vae_output_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 第二层：学习更复杂的空间特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        '''
        # 第三层（可选）：进一步精细化特征
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        '''
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1) 

    def forward(self, x):
        # x 预期形状为 (B*T, 3, H, W)，来自 VAE 解码器的输出
        B_original, C, T_output, H_out, W_out = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B_original * T_output, C, H_out, W_out)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_conv(x).view(B_original, T_output, 1, H_out, W_out).transpose(1, 2)
        # x = self.sigmoid(x)
        return x # 输出形状为 (B*T, 1, H, W)


    
class MaskOutputHead_3d(nn.Module):
    def __init__(self, vae_output_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(vae_output_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True)
        )

        # 第二层：学习更复杂的空间特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        '''
        # 第三层（可选）：进一步精细化特征
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        '''
        self.final_conv = nn.Conv3d(32, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)) 

    def forward(self, x):
        # x 预期形状为 (B*T, 3, H, W)，来自 VAE 解码器的输出
        B_original, C, T_output, H_out, W_out = x.shape
        x = self.conv1(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(B_original * T_output, 64, H_out, W_out)
        
        x = self.conv2(x)
        x = self.final_conv(x.view(B_original, T_output, 32, H_out, W_out).transpose(1, 2)).transpose(1, 2)
        # x = self.sigmoid(x)
        return x # 输出形状为 (B*T, 1, H, W)
    
class MaskVAEFinetuner(nn.Module):
    """
    A wrapper around Wan VAE for finetuning its decoder to output single-channel masks.
    """
    def __init__(self, args, vae_model_id: 'Wan2.1-T2V-1.3B-Diffusers_download', freeze_encoder: bool = True,
                 modify_type: str = 'replace_last_conv', target_dtype=torch.bfloat16):
        super().__init__()
        print(f"Loading VAE from {vae_model_id} (subfolder 'vae')...")
        self.vae = AutoencoderKLWan.from_pretrained(vae_model_id, subfolder="vae", torch_dtype=target_dtype)
        print("VAE loaded.")

        self.freeze_encoder = freeze_encoder
        self.modify_type = modify_type

        original_conv_out = self.vae.decoder.conv_out
        in_channels = original_conv_out.in_channels
        out_dim = in_channels


        new_conv_out = WanCausalConv3d(
            in_channels=out_dim,
            out_channels=1,      # <--- 核心改变：输出1个通道
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
            
        self.vae.encoder.eval() # 将 VAE 设置为评估模式，这对 BatchNorm/Dropout 层很重要


    def forward(self, mask_input: torch.Tensor):
        """
        Forward pass for mask VAE finetuning.
        Args:
            mask_input (torch.Tensor): Input masks, expected to be (B, T, 1, H, W).
                                        Will be converted to (B*T, 3, H, W) for VAE encoder.
        Returns:
            reconstructed_mask_logits (torch.Tensor): Reconstructed mask logits, (B, T, 1, H, W).
        """
        original_shape = mask_input.shape # B, T, 1, H, W

        # Convert to 3-channel (simulating RGB for VAE encoder) and flatten BxT
        # B, T, 1, H, W -> B, T, 3, H, W -> B*T, 3, H, W
        mask_input_rgb = mask_input.repeat(1, 1, 3, 1, 1).transpose(1, 2) # Repeat channel dim

        # Normalize to [-1, 1] if VAE expects it (Wan VAE typically does)
        # Assuming mask_input is [0, 1], so 2*x - 1
        mask_input_flat = mask_input_rgb * 2.0 - 1.0 

        # Encode
        # Returns LatentDist for KLD loss if training VAE from scratch, but for inference/finetuning
        # we can just sample from it or take the 'mean' for deterministic path.
        # For finetuning, it's typically fine to just use .sample() or .mean()
        # Ensure vae.encode returns a LatentDistribution object as per diffusers AutoencoderKL
        with torch.no_grad():
            latent_dist = self.vae.encode(mask_input_flat.to(self.vae.dtype)).latent_dist
        mask_latent = latent_dist.sample() # or latent_dist.mean() for deterministic
        
        # Decode
        # decoded_pixels will be (B*T, 1, H_out, W_out)
        reconstructed_mask_logits = self.vae.decode(mask_latent, return_dict=False)[0]

        #reconstructed_mask_logits = self.mask_output_head(reconstructed_rgb_from_vae.detach())
        return reconstructed_mask_logits

    def get_trainable_parameters(self):
        """
        Returns parameters that require gradients, specifically for finetuning.
        """
        return [p for p in self.parameters() if p.requires_grad]

