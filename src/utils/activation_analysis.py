"""
Activation Analysis for Neural Networks

This module provides functionality to analyze activations in neural networks,
particularly focused on the LightFormer model. It helps identify potential issues
such as vanishing or exploding activations, which can impact model performance
and training stability.

Key Features:
- Register hooks to capture activations during forward pass
- Compute and store activation statistics for each layer
- Visualize activation distributions using histograms
- Detect potential activation issues and provide warnings

Usage:
1. Import the module and create a LightFormer model instance
2. Run the analysis by calling the functions in this module
3. Examine the printed statistics and generated histograms

Functions:
- hook_fn(module, input, output): Hook function to capture layer outputs
- analyze_activations(activation_stats): Analyze and visualize activation statistics

Dependencies:
- torch
- numpy
- matplotlib
- models (custom module containing LightFormer)
- util (custom module containing run_with_animation)

Note:
This module is designed for use during the development and debugging phase of
neural network training, particularly for the LightFormer architecture. It may
introduce some computational overhead and should be used judiciously in
production environments.

Example:
    from models import LightFormer
    model = LightFormer()
    activation_stats = {}
    for name, module in model.named_modules():
        module.register_forward_hook(hook_fn)
    sample_input = torch.randn(16, 10, 3, 512, 960)
    run_with_animation(model, args=(sample_input,), name="Run Model")
    analyze_activations(activation_stats)

Author: Gordon Liu
Date: Oct 13 2024
Version: 1.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models import LightFormer
from threading import run_with_animation

def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        activation_stats[module] = output.detach().cpu().numpy()
    elif isinstance(output, tuple):
        activation_stats[module] = output[0].detach().cpu().numpy()

def analyze_activations(activation_stats):
    for module, activations in activation_stats.items():
        if isinstance(activations, np.ndarray):
            flat_activations = activations.flatten()

            # Basic statistics
            mean = np.mean(flat_activations)
            std = np.std(flat_activations)
            max_val = np.max(flat_activations)
            min_val = np.min(flat_activations)

            print(f"Module: {module}")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std Dev: {std:.4f}")
            print(f"  Max: {max_val:.4f}")
            print(f"  Min: {min_val:.4f}")

            # Check for potential issues
            if std < 1e-3:
                print("  WARNING: Very small standard deviation. Potential vanishing activations.")
            if std > 1e3:
                print("  WARNING: Very large standard deviation. Potential exploding activations.")
            if np.isnan(mean) or np.isinf(mean):
                print("  WARNING: NaN or Inf values detected.")

            # Plot histogram
            plt.figure(figsize=(10, 4))
            plt.hist(flat_activations, bins=50)
            plt.title(f"Activation Distribution for {module}")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.show()

            print("\n")

model = LightFormer()
activation_stats = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_fn)
sample_input = torch.randn(16, 10, 3, 512, 960)
run_with_animation(model, args=(sample_input,), name="Run Model")
analyze_activations(activation_stats)
