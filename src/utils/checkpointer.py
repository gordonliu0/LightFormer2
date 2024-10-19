import os
import torch
from datetime import datetime
from typing import Dict, Any
from utils.lr_scheduler import WarmupCosineScheduler

class ModelCheckpointer:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self,
                        epoch: int,
                        global_step: int,
                        model: torch.nn.Module,
                        optimizer: torch.optim.optimizer.Optimizer,
                        scheduler: WarmupCosineScheduler,
                        loss: float) -> None:
        """
        Save a checkpoint of the model. Checkpoints should save states ready for the next training epoch.

        Args:
            epoch: Next epoch of training.
            global_step: Next global step of training.
            model: Current model state.
            optimizer: Current optimizer state.
            scheduler: Current scheduler state.
            loss: Validation loss from last validation step.
        """
        # Strings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"epoch{epoch}time{timestamp}"
        model_path = os.path.join(self.save_dir, name)

        # Checkpoint Object
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'timestamp': timestamp,
            'loss': loss,
        }

        # Save
        torch.save(checkpoint, model_path)

    def remove_checkpoint(self, name) -> None:
        """Remove a checkpoint by name"""
        os.remove(os.path.join(self.save_dir, name))


# Example usage in a training loop:
# checkpointer = ModelCheckpointer('checkpoints', max_saves=5)
#
# for epoch in range(num_epochs):
#     for step, batch in enumerate(dataloader):
#         # ... training code ...
#
#         if step % save_interval == 0:
#             metric = evaluate_model(model, val_dataloader)
#             checkpointer.save_checkpoint(model, epoch, metric)
