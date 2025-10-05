import torch
import torch.nn as nn
from typing import List, Union, Optional, Tuple
from transformers import AutoTokenizer, UMT5EncoderModel
import html
import regex as re


def prompt_clean(text: str) -> str:
    """
    Cleans the prompt text by removing multiple spaces, HTML entities,
    and specific characters.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = html.unescape(text)
    text = re.sub(r'["\\]+', '', text) 
    return text


class TextProcessor:
    def __init__(self, tokenizer: AutoTokenizer, text_encoder: UMT5EncoderModel):
        """
        Initializes the TextProcessor.

        Args:
            tokenizer: The Hugging Face tokenizer (e.g., AutoTokenizer.from_pretrained(..., subfolder="tokenizer")).
            text_encoder: The Hugging Face text encoder model (e.g., UMT5EncoderModel.from_pretrained(..., subfolder="text_encoder")).
        """
        '''
        if not isinstance(tokenizer, AutoTokenizer) or not isinstance(text_encoder, UMT5EncoderModel):
            raise TypeError("tokenizer must be an AutoTokenizer instance and text_encoder a UMT5EncoderModel instance.")
        '''
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # Ensure text_encoder is in evaluation mode and its parameters are frozen.
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        print("TextProcessor initialized: Text Encoder set to eval mode and parameters frozen.")

    @torch.no_grad() # No need to track gradients for this frozen part
    def get_embeds_and_masks(
        self,
        prompt_list: List[str], # 接收一个字符串列表
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a list of prompts into text embeddings (encoder_hidden_states)
        and their corresponding attention masks.

        Args:
            prompt_list (List[str]): A list of text prompts.
            device (torch.device): The target device for the embeddings.
            dtype (torch.dtype): The target data type for the embeddings.
            max_sequence_length (int): Maximum sequence length for tokenization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - encoder_hidden_states (torch.Tensor): Encoded text embeddings.
                                                       Shape (batch_size, max_sequence_length, hidden_size).
                - attention_mask (torch.Tensor): Attention mask for the embeddings.
                                                 Shape (batch_size, max_sequence_length).
        """

        cleaned_prompt_list = [prompt_clean(p) for p in prompt_list]

        # Tokenization
        text_inputs = self.tokenizer(
            cleaned_prompt_list,
            padding="max_length", 
            max_length=max_sequence_length,
            truncation=True,    
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        self.text_encoder.eval()
        
        encoder_hidden_states = self.text_encoder(input_ids, attention_mask).last_hidden_state
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)

        return encoder_hidden_states, attention_mask
    
    @torch.no_grad() 
    def encode_prompt_and_cfg(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 64,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = False,
        num_videos_per_prompt: int = 1, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input prompt(s) and optionally negative prompt(s) into text embeddings
        for Classifier-Free Guidance (CFG).

        This method encapsulates the original _get_t5_prompt_embeds logic.

        Args:
            prompt (Union[str, List[str]]): The input text prompt or a list of prompts.
            device (Optional[torch.device]): The target device for the embeddings. If None, uses text_encoder's device.
            dtype (Optional[torch.dtype]): The target data type for the embeddings. If None, uses text_encoder's dtype.
            max_sequence_length (int): Maximum sequence length for tokenization.
            negative_prompt (Optional[Union[str, List[str]]]): Optional negative prompt(s).
            do_classifier_free_guidance (bool): Whether to perform CFG.
            num_videos_per_prompt (int): Number of video samples per prompt (for batch repetition).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prompt_embeds (torch.Tensor): Encoded text embeddings for the positive prompt.
                                                Shape (batch_size * num_videos_per_prompt, max_sequence_length, hidden_size).
                - negative_prompt_embeds (torch.Tensor): Encoded text embeddings for the negative prompt.
                                                          Shape (batch_size * num_videos_per_prompt, max_sequence_length, hidden_size).
                                                          This will be a tensor of zeros if CFG is not enabled.
        """
        device = device if device is not None else self.text_encoder.device
        dtype = dtype if dtype is not None else self.text_encoder.dtype

        # Handle positive prompt(s)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(p) for p in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, attention_mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = attention_mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        processed_prompt_embeds = []
        for u, v in zip(prompt_embeds, seq_lens):
            current_embed = u[:v] 
            if current_embed.size(0) < max_sequence_length:
                padding_size = max_sequence_length - current_embed.size(0)
                padding = current_embed.new_zeros(padding_size, current_embed.size(1))
                current_embed = torch.cat([current_embed, padding], dim=0)
            elif current_embed.size(0) > max_sequence_length:
                current_embed = current_embed[:max_sequence_length] 
            processed_prompt_embeds.append(current_embed)
        prompt_embeds = torch.stack(processed_prompt_embeds, dim=0)

        if do_classifier_free_guidance and negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif not do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            prompt_embeds = prompt_embeds.repeat(num_videos_per_prompt, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_videos_per_prompt, 1, 1)
            return prompt_embeds, negative_prompt_embeds

        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt = [prompt_clean(p) for p in negative_prompt]

        uncond_text_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        uncond_input_ids, uncond_attention_mask = uncond_text_inputs.input_ids, uncond_text_inputs.attention_mask
        uncond_seq_lens = uncond_attention_mask.gt(0).sum(dim=1).long()

        negative_prompt_embeds = self.text_encoder(uncond_input_ids.to(device), uncond_attention_mask.to(device)).last_hidden_state
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)

        processed_negative_prompt_embeds = []
        for u, v in zip(negative_prompt_embeds, uncond_seq_lens):
            current_embed = u[:v]
            if current_embed.size(0) < max_sequence_length:
                padding_size = max_sequence_length - current_embed.size(0)
                padding = current_embed.new_zeros(padding_size, current_embed.size(1))
                current_embed = torch.cat([current_embed, padding], dim=0)
            elif current_embed.size(0) > max_sequence_length:
                current_embed = current_embed[:max_sequence_length]
            processed_negative_prompt_embeds.append(current_embed)
        negative_prompt_embeds = torch.stack(processed_negative_prompt_embeds, dim=0)

        prompt_embeds = prompt_embeds.repeat(num_videos_per_prompt, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(num_videos_per_prompt, 1, 1)

        return prompt_embeds, negative_prompt_embeds

