import torch
import math
import numpy as np
import scipy.stats
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
#import seaborn as sns
import os

WAN_PRETRAINED_TOTAL_TIMESTEPS = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import torch
import numpy as np
from typing import List, Tuple, Optional


def boundary_biased_sample(
    total_train_timesteps: int,
    num_samples_per_batch: int, 
    sampling_strategy: str = "uniform",
    bias_param: float = 1.0,
    special_timesteps: Optional[List[int]] = None,
    special_freq: float = 0.1, 
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates custom timesteps for training, allowing control over specific timestep frequencies.
    Modified to handle num_samples_per_batch = 1 correctly for probability control.

    Args:
        total_train_timesteps (int): The total number of diffusion steps used during training (e.g., 1000).
        num_samples_per_batch (int): The total number of timesteps to generate for the current batch.
        sampling_strategy (str): The base strategy to sample timesteps ("uniform", "bias_video", "bias_mask", "karras").
        bias_param (float): Parameter to control the degree of bias for "bias_video", "bias_mask".
        special_timesteps (Optional[List[int]]): List of specific integer timesteps whose frequency needs to be controlled.
                                                If None, no special frequency control is applied.
        special_freq (float): The proportion of the batch that should be dedicated to `special_timesteps`.
                              e.g., 0.1 means 10% of `num_samples_per_batch` will be from `special_timesteps`.
        device (torch.device): The device to place the tensors on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - timestep_for_dit (torch.Tensor): Integer timesteps (shape: [batch_size]).
            - t_expanded (torch.Tensor): Continuous time embeddings (shape: [batch_size, 1, 1, 1, 1]).
    """

    if special_timesteps is not None and not isinstance(special_timesteps, list):
        special_timesteps = [special_timesteps] # Ensure it's a list

    # --- Decision logic for generating timesteps ---
    
    # Validate special_timesteps first
    valid_special_timesteps = []
    if special_timesteps:
        valid_special_timesteps = [
            ts for ts in special_timesteps if 1 <= ts <= total_train_timesteps
        ]
        if len(valid_special_timesteps) < len(special_timesteps):
            print(f"Warning: Some special timesteps were out of range [1, {total_train_timesteps}] and were ignored.")
        special_timesteps = valid_special_timesteps

    # --- Case 1: num_samples_per_batch is 1 ---
    if num_samples_per_batch == 1:
        # Use special_freq as a direct probability for this single sample
        if special_timesteps and torch.rand(1, device=device).item() < special_freq:
            # Choose a random special timestep
            if not special_timesteps: # Should not happen due to validation above, but for safety
                print("Warning: special_freq > 0 but special_timesteps became empty. Falling back to uniform.")
                chosen_timestep_int = torch.randint(1, total_train_timesteps + 1, (1,), device=device, dtype=torch.long)
            else:
                special_ts_tensor = torch.tensor(special_timesteps, device=device, dtype=torch.long)
                chosen_timestep_int = special_ts_tensor[torch.randint(0, len(special_ts_tensor), (1,), device=device)]
        else:
            # Choose a regular timestep based on the specified sampling_strategy
            # We need a helper function or inline logic to do this for a single sample.
            # For simplicity, we'll use uniform sampling here. If you need bias, you'd adapt that part.
            if sampling_strategy == "uniform":
                chosen_timestep_int = torch.randint(1, total_train_timesteps + 1, (1,), device=device, dtype=torch.long)
            elif sampling_strategy in ["bias_video", "bias_mask"]:
                # Apply bias logic for a single sample
                beta = bias_param
                power = 1.0 / (beta + 1.0)
                u_sample = torch.rand(1, device=device)
                if sampling_strategy == "bias_video":
                    continuous_timestep = u_sample ** power
                else: # bias_mask
                    continuous_timestep = 1.0 - (u_sample ** power)
                continuous_timestep = continuous_timestep.clamp(1e-5, 1.0)
                chosen_timestep_int = (continuous_timestep * total_train_timesteps).long().clamp(1, total_train_timesteps)
            elif sampling_strategy == "karras":
                # Karras logic for a single sample
                rho = 7.0
                # We need to generate sigmas based on num_samples_per_batch=1
                # This means we need to generate a single sigma value.
                # The ramp generation needs adjustment to produce a single value.
                
                # A simplified Karras for single sample:
                # Generate one random number, map it to a sigma value in the Karras schedule.
                u = torch.rand(1, device=device)
                sigma_min_approx = 1.0 / total_train_timesteps
                sigma_max_approx = 1.0
                min_inv_rho = sigma_min_approx ** (1 / rho)
                max_inv_rho = sigma_max_approx ** (1 / rho)
                
                # Generate a single point within the inv_rho range
                inv_rho_val = max_inv_rho + u * (min_inv_rho - max_inv_rho)
                sigma_val = inv_rho_val ** rho
                sigma_val = torch.max(sigma_val, torch.tensor(1e-5, device=device))
                
                estimated_timestep = sigma_val * total_train_timesteps
                chosen_timestep_int = torch.round(estimated_timestep).long().clamp(1, total_train_timesteps)
            else:
                raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

        timestep_for_dit = chosen_timestep_int.to(device=device)

    # --- Case 2: num_samples_per_batch > 1 (Original logic) ---
    else:
        # Determine how many "regular" samples we need to generate
        num_special_samples = 0
        if special_timesteps and len(special_timesteps) > 0:
            num_special_samples = int(num_samples_per_batch * special_freq)
            num_special_samples = min(num_special_samples, num_samples_per_batch)
            num_regular_samples = num_samples_per_batch - num_special_samples
            if num_regular_samples < 0:
                num_regular_samples = 0
        else:
            num_regular_samples = num_samples_per_batch
            num_special_samples = 0

        # --- Generate the "regular" samples ---
        regular_timesteps_int = None
        if num_regular_samples > 0:
            if sampling_strategy == "uniform":
                all_possible_timesteps = torch.arange(1, total_train_timesteps + 1, device=device)
                indices = torch.randperm(total_train_timesteps, device=device)[:num_regular_samples]
                regular_timesteps_int = all_possible_timesteps[indices]

            elif sampling_strategy in ["bias_video", "bias_mask"]:
                beta = bias_param
                power = 1.0 / (beta + 1.0)
                u_samples = torch.rand(num_regular_samples, device=device)
                if sampling_strategy == "bias_video":
                    continuous_timesteps = u_samples ** power
                else: # bias_mask
                    continuous_timesteps = 1.0 - (u_samples ** power)
                continuous_timesteps = continuous_timesteps.clamp(1e-5, 1.0)
                regular_timesteps_int = (continuous_timesteps * total_train_timesteps).long().clamp(1, total_train_timesteps)

            elif sampling_strategy == "karras":
                rho = 7.0
                ramp = np.linspace(0, 1, num_regular_samples)
                sigma_min_approx = 1.0 / total_train_timesteps
                sigma_max_approx = 1.0
                min_inv_rho = sigma_min_approx ** (1 / rho)
                max_inv_rho = sigma_max_approx ** (1 / rho)
                karras_sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
                karras_sigmas = np.maximum(karras_sigmas, 1e-5)
                estimated_timesteps = karras_sigmas * total_train_timesteps
                regular_timesteps_int = np.round(estimated_timesteps).astype(np.long)
                regular_timesteps_int = np.clip(regular_timesteps_int, 1, total_train_timesteps)
                regular_timesteps_int = torch.from_numpy(regular_timesteps_int).to(device=device)
            else:
                raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

        # --- Generate the "special" samples ---
        special_timesteps_int = None
        if num_special_samples > 0 and special_timesteps and len(special_timesteps) > 0:
            special_ts_tensor = torch.tensor(special_timesteps, device=device, dtype=torch.long)
            indices_for_special = torch.randint(0, len(special_ts_tensor), (num_special_samples,), device=device)
            special_timesteps_int = special_ts_tensor[indices_for_special]

        # --- Combine regular and special samples ---
        if num_regular_samples > 0 and num_special_samples > 0:
            timestep_for_dit = torch.cat((regular_timesteps_int, special_timesteps_int), dim=0)
        elif num_regular_samples > 0:
            timestep_for_dit = regular_timesteps_int
        elif num_special_samples > 0:
            timestep_for_dit = special_timesteps_int
        else:
            timestep_for_dit = torch.empty(0, dtype=torch.long, device=device)

        # Ensure the final output has the correct batch_size, and shuffle if needed
        if len(timestep_for_dit) < num_samples_per_batch:
            needed_samples = num_samples_per_batch - len(timestep_for_dit)
            fallback_samples = torch.randint(1, total_train_timesteps + 1, (needed_samples,), device=device, dtype=torch.long)
            timestep_for_dit = torch.cat((timestep_for_dit, fallback_samples), dim=0)

        if len(timestep_for_dit) > 1:
            shuffle_indices = torch.randperm(len(timestep_for_dit), device=device)
            timestep_for_dit = timestep_for_dit[shuffle_indices]
        
        timestep_for_dit = timestep_for_dit[:num_samples_per_batch]

    # --- Convert integer timesteps to continuous t and expanded t ---
    t_continuous = timestep_for_dit.float() / total_train_timesteps
    t_continuous = t_continuous.clamp(1e-5, 1.0)
    
    t_expanded = t_continuous.view(-1, 1, 1, 1, 1)
    
    return timestep_for_dit, t_expanded
