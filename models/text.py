import torch
import torch.nn as nn
from typing import List, Union, Optional, Tuple
from transformers import AutoTokenizer, UMT5EncoderModel
import html
import regex as re

# --- 辅助函数：根据你 MyWanPipeline 原始代码中的 prompt_clean 逻辑 ---
def prompt_clean(text: str) -> str:
    """
    Cleans the prompt text by removing multiple spaces, HTML entities,
    and specific characters.
    """
    # This is a direct extraction of the prompt_clean logic from MyWanPipeline.
    # It removes multiple spaces, decodes HTML entities, and removes specific characters.
    text = re.sub(r'\s+', ' ', text).strip()
    text = html.unescape(text)
    text = re.sub(r'["\\]+', '', text) # Remove double quotes and backslashes
    return text

# --- TextProcessor 类 ---
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
        # 清理每个 prompt
        cleaned_prompt_list = [prompt_clean(p) for p in prompt_list]

        # Tokenization
        text_inputs = self.tokenizer(
            cleaned_prompt_list,
            padding="max_length", # 填充到 max_length
            max_length=max_sequence_length,
            truncation=True,    # 截断到 max_length
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # 文本编码
        # 确保 text_encoder 处于 eval 模式且参数已冻结
        # (这应该已经在 __init__ 中处理，这里是再次确认)
        self.text_encoder.eval()
        
        encoder_hidden_states = self.text_encoder(input_ids, attention_mask).last_hidden_state
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)

        # 经过 tokenizer 的 padding="max_length" 和 truncation=True，
        # encoder_hidden_states 和 attention_mask 的形状应该已经是 (batch_size, max_sequence_length, hidden_size) 和 (batch_size, max_sequence_length)。
        # 因此，不需要像 original encode_prompt_and_cfg 中那样额外的循环处理。

        return encoder_hidden_states, attention_mask
    
    @torch.no_grad() # No need to track gradients for this frozen part
    def encode_prompt_and_cfg(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 64,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = False,
        num_videos_per_prompt: int = 1, # Typically 1 for RVOS
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

        # Pad or truncate embeddings to max_sequence_length
        processed_prompt_embeds = []
        for u, v in zip(prompt_embeds, seq_lens):
            current_embed = u[:v] # Take actual sequence length
            if current_embed.size(0) < max_sequence_length:
                padding_size = max_sequence_length - current_embed.size(0)
                padding = current_embed.new_zeros(padding_size, current_embed.size(1))
                current_embed = torch.cat([current_embed, padding], dim=0)
            elif current_embed.size(0) > max_sequence_length:
                current_embed = current_embed[:max_sequence_length] # Should not happen with truncation=True
            processed_prompt_embeds.append(current_embed)
        prompt_embeds = torch.stack(processed_prompt_embeds, dim=0)

        # Handle negative prompt(s) for CFG
        if do_classifier_free_guidance and negative_prompt is None:
            # If CFG is enabled but negative_prompt is None, use empty string for unconditional context
            negative_prompt = [""] * batch_size
        elif not do_classifier_free_guidance:
            # If CFG is disabled, negative_prompt_embeds is just zeros
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            # Repeat for num_videos_per_prompt if needed for consistency with batch size expectation
            prompt_embeds = prompt_embeds.repeat(num_videos_per_prompt, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_videos_per_prompt, 1, 1)
            return prompt_embeds, negative_prompt_embeds

        # Process negative prompt if CFG is enabled
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

        # Pad or truncate negative embeddings
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

        # Repeat for num_videos_per_prompt for both positive and negative
        prompt_embeds = prompt_embeds.repeat(num_videos_per_prompt, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(num_videos_per_prompt, 1, 1)

        return prompt_embeds, negative_prompt_embeds

# --- 使用示例 (在你的 main.py 或其他地方) ---
if __name__ == '__main__':
    # Mocking Hugging Face components for demonstration
    class MockTokenizer:
        def __call__(self, text, padding, max_length, truncation, add_special_tokens, return_attention_mask, return_tensors):
            if isinstance(text, str):
                text = [text]
            # Simulate tokenization with varying lengths for testing padding
            ids = []
            masks = []
            for t in text:
                token_ids = list(range(1, min(max_length, len(t) + 1))) # Simple mock
                ids.append(token_ids + [0] * (max_length - len(token_ids)))
                masks.append([1] * len(token_ids) + [0] * (max_length - len(token_ids)))
            
            return type('MockOutput', (object,), {
                'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor(masks)
            })()

    class MockTextEncoder(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.linear = nn.Linear(5, hidden_size) # Input dim (mocked token_ids) to hidden_size
            self.eval()
            self.requires_grad_(False)
            self._dtype = torch.float32 # Set a default dtype for the mock

        @property
        def device(self):
            return next(self.parameters()).device
        
        @property
        def dtype(self):
            return self._dtype

        def forward(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            # Create mock output with correct dimensions and random values
            mock_output = torch.randn(batch_size, seq_len, self.linear.out_features)
            return type('MockEncoderOutput', (object,), {'last_hidden_state': mock_output})()

    mock_tokenizer = MockTokenizer()
    mock_text_encoder = MockTextEncoder()

    text_processor = TextProcessor(mock_tokenizer, mock_text_encoder)

    # Test Case 1: Single prompt, no CFG (training scenario)
    prompt = "A cat on the sofa"
    prompt_embeds, negative_prompt_embeds = text_processor.encode_prompt_and_cfg(
        prompt=prompt, 
        do_classifier_free_guidance=False, 
        max_sequence_length=10,
        device=torch.device("cpu"),
        dtype=torch.float32
    )
    print("\n--- Test Case 1: Single prompt, no CFG ---")
    print(f"Prompt Embeds shape: {prompt_embeds.shape}") # Expected: (1, 10, 768)
    print(f"Negative Prompt Embeds shape: {negative_prompt_embeds.shape}") # Expected: (1, 10, 768) and all zeros
    print(f"Is negative_prompt_embeds all zeros? {torch.all(negative_prompt_embeds == 0.0)}")

    # Test Case 2: Batch prompts, no CFG
    prompts = ["A dog running in the park", "A bird flying in the sky"]
    prompt_embeds_batch, negative_prompt_embeds_batch = text_processor.encode_prompt_and_cfg(
        prompt=prompts, 
        do_classifier_free_guidance=False, 
        max_sequence_length=10,
        device=torch.device("cpu"),
        dtype=torch.float32
    )
    print("\n--- Test Case 2: Batch prompts, no CFG ---")
    print(f"Prompt Embeds Batch shape: {prompt_embeds_batch.shape}") # Expected: (2, 10, 768)
    print(f"Negative Prompt Embeds Batch shape: {negative_prompt_embeds_batch.shape}") # Expected: (2, 10, 768) and all zeros
    print(f"Is negative_prompt_embeds_batch all zeros? {torch.all(negative_prompt_embeds_batch == 0.0)}")


    # Test Case 3: Single prompt, with CFG and negative prompt
    prompt_cfg = "A happy dog"
    negative_prompt_cfg = "A sad cat"
    prompt_embeds_cfg, negative_prompt_embeds_cfg = text_processor.encode_prompt_and_cfg(
        prompt=prompt_cfg,
        negative_prompt=negative_prompt_cfg,
        do_classifier_free_guidance=True,
        max_sequence_length=10,
        device=torch.device("cpu"),
        dtype=torch.float32
    )
    print("\n--- Test Case 3: Single prompt, with CFG and negative prompt ---")
    print(f"Prompt Embeds CFG shape: {prompt_embeds_cfg.shape}") # Expected: (1, 10, 768)
    print(f"Negative Prompt Embeds CFG shape: {negative_prompt_embeds_cfg.shape}") # Expected: (1, 10, 768) and NOT all zeros
    print(f"Is negative_prompt_embeds_cfg all zeros? {torch.all(negative_prompt_embeds_cfg == 0.0)}") # Should be False

    # Test Case 4: Single prompt, with CFG but no explicit negative prompt (uses empty string)
    prompt_cfg_no_neg = "A smiling person"
    prompt_embeds_cfg_no_neg, negative_prompt_embeds_cfg_no_neg = text_processor.encode_prompt_and_cfg(
        prompt=prompt_cfg_no_neg,
        negative_prompt=None, # Explicitly None
        do_classifier_free_guidance=True,
        max_sequence_length=10,
        device=torch.device("cpu"),
        dtype=torch.float32
    )
    print("\n--- Test Case 4: Single prompt, with CFG, no explicit negative prompt ---")
    print(f"Prompt Embeds CFG (no neg) shape: {prompt_embeds_cfg_no_neg.shape}") # Expected: (1, 10, 768)
    print(f"Negative Prompt Embeds CFG (no neg) shape: {negative_prompt_embeds_cfg_no_neg.shape}") # Expected: (1, 10, 768) and all zeros (empty string embeds to zeros)
    print(f"Is negative_prompt_embeds_cfg_no_neg all zeros? {torch.all(negative_prompt_embeds_cfg_no_neg == 0.0)}") # Should be True because "" maps to near-zero embeddings
