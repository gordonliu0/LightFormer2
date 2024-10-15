import os
import torch
import datetime
from typing import Dict, Any

class ModelCheckpointer:
    def __init__(self, save_dir: str, max_saves: int = 3):
        self.save_dir = save_dir
        self.max_saves = max_saves
        self.checkpoints = []
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, model: torch.nn.Module, epoch: int, metric: float) -> None:
        """Save a checkpoint of the model."""
        # Strings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"t_{timestamp}_e_{epoch}"
        model_path = os.path.join(self.save_dir, name)

        # Save
        self.checkpoints.append = (name, metric)
        torch.save(model.state_dict(), model_path)

        # Remove old checkpoints if exceeding max_saves
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the latest max_saves."""
        self.checkpoints.sort(key=lambda x: x[1], reverse=True)
        for ckpt in self.checkpoints[self.max_saves:]:
            os.remove(os.path.join(self.save_dir, ckpt[0]))

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
