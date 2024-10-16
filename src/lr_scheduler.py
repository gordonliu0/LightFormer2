import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

class WarmupCosineScheduler():
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_start_factor: float = 0.0,
        warmup_end_factor: float = 1.0,
        T_0 : int = 1,
        T_mult : int = 2,
        eta_min: float = 0.0
    ):
        self.scheduler = self._create_warmup_cosine_scheduler(optimizer, warmup_steps, warmup_start_factor, warmup_end_factor, T_0, T_mult, eta_min)

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def _create_warmup_cosine_scheduler(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_start_factor: float = 0.0,
        warmup_end_factor: float = 1.0,
        T_0 : int = 1,
        T_mult : int = 2,
        eta_min: float = 0.0
    ) -> torch.optim.lr_scheduler.SequentialLR:
        """
        Creates a learning rate scheduler with linear warmup and cosine annealing with warm restarts until the end.

        Args:
        optimizer: The optimizer for which to schedule the learning rate.
        warmup_epochs: Number of warmup epochs.
        warmup_start_factor: The fraction of initial_lr to start the warmup from. Default: 0.0
        warmup_start_factor: The fraction of initial_lr to end the warmup at. Default: 1.0
        T_0: Number of iterations until the first restart after the warmup period. Default: 1
        T_mult: A factor by which T_i increases after a restart. Default: 2
        eta_min: Minimum learning rate reached at the end of each restart cycle. Default: 0.0

        Returns:
        A PyTorch SequentialLR scheduler
        """
        # Linear warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=warmup_end_factor,
            total_iters=warmup_steps
        )

        # Cosine decay scheduler
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )

        # Combine schedulers
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

def main():
    # Define a simple model (for demonstration purposes)
    model = torch.nn.Linear(10, 1)

    # Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Create a learning rate scheduler
    scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_steps=4, warmup_start_factor=0.001, warmup_end_factor=1, T_0=1, T_mult=2)

    # Number of epochs
    num_epochs = 32

    # List to store learning rates
    lrs = []

    # Simulate training loop
    for epoch in range(num_epochs):
        # Your training code would go here

        # Step the scheduler
        scheduler.step()

        # Append the current learning rate
        lrs.append(scheduler.get_last_lr()[0])  # get_last_lr() returns a list, so we take the first element

    # Plot the learning rates
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid()
    plt.title('Learning Rate vs. Epoch')
    plt.show()

main()
