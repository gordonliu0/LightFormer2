import os
import torch
from datetime import datetime
from utils.lr_scheduler import WarmupCosineScheduler

class ModelCheckpointer:
    def __init__(self, save_dir: str, max_checkpoints: int = 10, keep_latest: bool = True, prune_fn: function = lambda x: x['loss'], prune_lowest: bool = False):
        """
        Checkpointer controlling checkpoint behavior of training.
        Checkpointer keeps max_checkpoints 'best' checkpoints based on prune_fn. By default, keeps 5 based on lowest loss.
        If keep_latest = True, will keep max_checkpoints-1 of the top checkpoints AND latest.

        Args:
            save_dir: Directory holding checkpoints.
            max_checkpoints: Most checkpoints to keep saved at one time.
            keep_latest: Whether the latest checkpoint should always be kept. Useful for continuing interrupted training.
            prune_fn: The function to use for checkpoint pruning.
            prune_lowest: By default, pruning keeps the checkpoints with the lowest values returned from prune_fn. If True, pruning keeps checkpoints with highest values.
        """
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._max_checkpoints = max_checkpoints
        self._checkpoints = [self.load_checkpoint(n) for n in os.listdir(self.save_dir)]
        self._keep_latest = keep_latest
        self._prune_fn = prune_fn
        self._prune_lowest = prune_lowest

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def max_checkpoints(self):
        return self._max_checkpoints

    @property
    def checkpoints(self):
        return self._checkpoints

    @checkpoints.setter
    def checkpoints(self, value):
        self._checkpoints = value

    def __len__(self):
        return len(self.checkpoints)

    def save_checkpoint(self,
                        epoch: int,
                        global_step: int,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
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
        path = os.path.join(self.save_dir, name)

        # Checkpoint Object
        checkpoint = {
            'path': path,
            'name': name,
            'epoch': epoch,
            'global_step': global_step,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'timestamp': timestamp,
            'loss': loss,
        }

        if self._keep_latest: # first prune then add
            self.prune_checkpoints(self.max_checkpoints-1)
            self.checkpoints.append(checkpoint)
            torch.save(checkpoint, path)
        else: # first add then prune
            self.checkpoints.append(checkpoint)
            torch.save(checkpoint, path)
            self.prune_checkpoints(self.max_checkpoints)

    def prune_checkpoints(self, keep):
        """
        Prunes checkpoints by checkpointer properties.
        """
        self.checkpoints.sort(key=self._prune_fn, reverse=self._prune_lowest)
        while len(self.checkpoints) > keep:
            removed = self.checkpoints.pop()
            self.remove_checkpoint_by_name(removed['name'])

    def remove_checkpoint(self, index: int) -> None:
        """Remove a checkpoint by index"""
        ckpt = self.checkpoints.pop(index)
        self.remove_checkpoint_by_name(ckpt['name'])

    def remove_checkpoint_by_name(self, name) -> None:
        """Remove a checkpoint by name"""
        os.remove(os.path.join(self.save_dir, name))

    def load_checkpoint_by_name(self, name: str):
        """Load a checkpoint by index."""
        checkpoint_path = os.path.join(self.save_dir, name)
        checkpoint = torch.load(checkpoint_path)
        return checkpoint

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
