import torch
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
        self.scheduler()

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

