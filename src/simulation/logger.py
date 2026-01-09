"""Simulation logging system."""

import json
import os
from typing import Dict, Any, List
from datetime import datetime


class SimulationLogger:
    """Logs simulation data to files.
    
    Records timestep-by-timestep data for later analysis.
    """
    
    def __init__(self, run_id: int, log_dir: str = "results"):
        """Initialize logger.
        
        Args:
            run_id: Unique run identifier
            log_dir: Directory for log files
        """
        self.run_id = run_id
        self.log_dir = log_dir
        
        # Create log directory
        self.run_dir = os.path.join(log_dir, f"run_{run_id:04d}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Log files
        self.timestep_log_path = os.path.join(self.run_dir, "timesteps.jsonl")
        self.summary_log_path = os.path.join(self.run_dir, "summary.json")
        
        # Buffers
        self.timestep_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100
        
        # Metadata
        self.start_time = datetime.now()
        self.log_count = 0
    
    def log_timestep(self, timestep: int, 
                     organism_states: List[Dict[str, Any]],
                     environment_state: Dict[str, Any],
                     collisions: List[Dict[str, Any]]):
        """Log data for one timestep.
        
        Args:
            timestep: Current timestep
            organism_states: States of all organisms
            environment_state: Environment state
            collisions: Collision events
        """
        log_entry = {
            'timestep': timestep,
            'organism_states': organism_states,
            'environment_state': environment_state,
            'collisions': collisions
        }
        
        self.timestep_buffer.append(log_entry)
        self.log_count += 1
        
        # Flush buffer if full
        if len(self.timestep_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffer to disk."""
        if not self.timestep_buffer:
            return
        
        with open(self.timestep_log_path, 'a') as f:
            for entry in self.timestep_buffer:
                f.write(json.dumps(entry) + '\n')
        
        self.timestep_buffer = []
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary statistics.
        
        Args:
            summary: Summary dictionary
        """
        # Add metadata
        summary['start_time'] = self.start_time.isoformat()
        summary['end_time'] = datetime.now().isoformat()
        summary['total_logs'] = self.log_count
        
        with open(self.summary_log_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def close(self):
        """Close logger and flush remaining data."""
        self.flush()
    
    def get_log_paths(self) -> Dict[str, str]:
        """Get paths to log files.
        
        Returns:
            Dictionary with log file paths
        """
        return {
            'timestep_log': self.timestep_log_path,
            'summary_log': self.summary_log_path,
            'run_dir': self.run_dir
        }
