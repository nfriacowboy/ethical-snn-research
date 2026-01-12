"""Simulation checkpointing for save/load."""

import os
import pickle
from typing import Any, Dict


class Checkpointer:
    """Saves and loads simulation checkpoints.

    Enables resuming simulations and analyzing intermediate states.
    """

    def __init__(self, run_id: int, checkpoint_dir: str = "results/checkpoints"):
        """Initialize checkpointer.

        Args:
            run_id: Unique run identifier
            checkpoint_dir: Directory for checkpoints
        """
        self.run_id = run_id
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        self.run_checkpoint_dir = os.path.join(checkpoint_dir, f"run_{run_id:04d}")
        os.makedirs(self.run_checkpoint_dir, exist_ok=True)

    def save(self, checkpoint_data: Dict[str, Any], timestep: int):
        """Save checkpoint.

        Args:
            checkpoint_data: Data to save
            timestep: Current timestep
        """
        checkpoint_path = os.path.join(
            self.run_checkpoint_dir, f"checkpoint_{timestep:06d}.pkl"
        )

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def load(self, timestep: int) -> Dict[str, Any]:
        """Load checkpoint.

        Args:
            timestep: Timestep to load

        Returns:
            Checkpoint data
        """
        checkpoint_path = os.path.join(
            self.run_checkpoint_dir, f"checkpoint_{timestep:06d}.pkl"
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        return checkpoint_data

    def list_checkpoints(self) -> list:
        """List available checkpoints.

        Returns:
            List of checkpoint timesteps
        """
        if not os.path.exists(self.run_checkpoint_dir):
            return []

        checkpoints = []
        for filename in os.listdir(self.run_checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".pkl"):
                timestep_str = filename.replace("checkpoint_", "").replace(".pkl", "")
                timestep = int(timestep_str)
                checkpoints.append(timestep)

        return sorted(checkpoints)

    def get_latest_checkpoint(self) -> int:
        """Get timestep of latest checkpoint.

        Returns:
            Latest checkpoint timestep, or -1 if none exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else -1

    def clean_old_checkpoints(self, keep_every: int = 5):
        """Remove old checkpoints, keeping only every Nth.

        Args:
            keep_every: Keep every Nth checkpoint
        """
        checkpoints = self.list_checkpoints()

        for i, timestep in enumerate(checkpoints):
            # Keep first, last, and every Nth
            if i == 0 or i == len(checkpoints) - 1 or i % keep_every == 0:
                continue

            # Delete this checkpoint
            checkpoint_path = os.path.join(
                self.run_checkpoint_dir, f"checkpoint_{timestep:06d}.pkl"
            )
            os.remove(checkpoint_path)
