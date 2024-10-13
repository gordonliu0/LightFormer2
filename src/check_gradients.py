import torch
import numpy as np
import matplotlib.pyplot as plt
from models import LightFormer
from util import run_with_animation

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
