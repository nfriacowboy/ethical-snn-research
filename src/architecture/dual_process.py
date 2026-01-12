"""Dual-Process Organism Architecture (Condition B).

Integrates SurvivalSNN (SNN-S) with EthicalSNN (SNN-E) to create organisms
that balance survival needs with ethical considerations.

Architecture:
    SNN-S proposes actions → SNN-E evaluates → Final action selected
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.organisms.base_organism import Organism, Action
from src.organisms.survival_snn import SurvivalSNN
from src.organisms.ethical_snn import EthicalSNN


class DualProcessOrganism(Organism):
    """Organism with dual-process architecture (SNN-S + SNN-E).
    
    This is Condition B in Phase 1 experiments.
    
    Decision process:
        1. SNN-S receives environment state and proposes action
        2. SNN-E evaluates proposed action in ethical context
        3. If action is ethical, execute it
        4. If action is unethical, select fallback action (WAIT)
    
    Attributes:
        survival_snn: Neural network for survival decisions
        ethical_snn: Neural network for ethical evaluation
        veto_count: Number of times ethical network vetoed actions
        approval_count: Number of times ethical network approved actions
    """
    
    def __init__(
        self,
        organism_id: int,
        position: Tuple[int, int],
        energy: float = 100.0,
        survival_hidden: int = 30,
        ethical_hidden: int = 20
    ):
        """Initialize dual-process organism.
        
        Args:
            organism_id: Unique identifier
            position: Initial (row, col) position
            energy: Initial energy level (default: 100.0)
            survival_hidden: Hidden layer size for SNN-S (default: 30)
            ethical_hidden: Hidden layer size for SNN-E (default: 20)
        """
        super().__init__(organism_id, position, energy)
        
        # Store initial values for reset
        self.initial_position = position
        self.initial_energy = energy
        
        # Survival network (SNN-S)
        self.survival_snn = SurvivalSNN(
            organism_id=organism_id,
            position=position,
            initial_energy=energy,
            input_size=8,
            hidden_size=survival_hidden,
            output_size=5
        )
        
        # Ethical network (SNN-E)
        self.ethical_snn = EthicalSNN(
            input_size=8,
            hidden_size=ethical_hidden,
            output_size=2
        )
        
        # Statistics
        self.veto_count = 0
        self.approval_count = 0
        self.fallback_count = 0
        
    def decide(self, state: Dict[str, Any]) -> Action:
        """Make decision using dual-process architecture.
        
        Process:
            1. SNN-S proposes action based on environment
            2. Extract ethical context from state
            3. SNN-E evaluates proposed action
            4. Return action if approved, else fallback to WAIT
        
        Args:
            state: Environment state dictionary with keys:
                - food_direction: (dx, dy) or None
                - energy: current energy level
                - obstacles_nearby: List of (dx, dy) obstacle directions
                - other_organism: (dx, dy, other_energy) or None
                - food_at_position: bool
                - grid_size: int
        
        Returns:
            Final action (Action enum)
        """
        # Step 1: SNN-S proposes action
        proposed_action = self.survival_snn.decide(state)
        
        # Step 2: Extract ethical context
        ethical_context = self._extract_ethical_context(state, proposed_action)
        
        # Step 3: SNN-E evaluates
        is_ethical = self.ethical_snn.evaluate_action(
            self_energy=ethical_context['self_energy'],
            other_energy=ethical_context['other_energy'],
            food_available=ethical_context['food_available'],
            distance_to_other=ethical_context['distance_to_other'],
            proposed_action=ethical_context['action_str'],
            num_steps=10
        )
        
        # Step 4: Apply decision
        if is_ethical:
            self.approval_count += 1
            final_action = proposed_action
        else:
            # Veto: fallback to WAIT
            self.veto_count += 1
            self.fallback_count += 1
            final_action = Action.WAIT
        
        # Log the decision
        self.log_action(final_action)
        
        return final_action
    
    def _extract_ethical_context(
        self,
        state: Dict[str, Any],
        proposed_action: Action
    ) -> Dict[str, Any]:
        """Extract ethical evaluation context from environment state.
        
        Args:
            state: Environment state
            proposed_action: Action proposed by SNN-S
        
        Returns:
            Dictionary with ethical context:
                - self_energy: float [0, 100]
                - other_energy: float [0, 100] (or 100 if no other organism)
                - food_available: bool
                - distance_to_other: float [0, 20]
                - action_str: str ('ATTACK', 'EAT', 'MOVE', 'WAIT')
        """
        # Self energy
        self_energy = state.get('energy', self.energy)
        
        # Other organism info
        other_organism = state.get('other_organism', None)
        if other_organism and len(other_organism) == 3:
            dx, dy, other_energy = other_organism
            distance_to_other = abs(dx) + abs(dy)  # Manhattan distance
        else:
            other_energy = 100.0  # Assume healthy if not present
            distance_to_other = 20.0  # Far away
        
        # Food availability
        food_available = state.get('food_at_position', False)
        if not food_available:
            # Check if food is nearby
            food_direction = state.get('food_direction', None)
            food_available = (food_direction is not None)
        
        # Action string
        action_map = {
            Action.ATTACK: 'ATTACK',
            Action.EAT: 'EAT',
            Action.WAIT: 'WAIT',
            Action.MOVE_NORTH: 'MOVE',
            Action.MOVE_SOUTH: 'MOVE',
            Action.MOVE_EAST: 'MOVE',
            Action.MOVE_WEST: 'MOVE'
        }
        action_str = action_map.get(proposed_action, 'WAIT')
        
        return {
            'self_energy': float(self_energy),
            'other_energy': float(other_energy),
            'food_available': bool(food_available),
            'distance_to_other': float(distance_to_other),
            'action_str': action_str
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get organism statistics including dual-process metrics.
        
        Returns:
            Dictionary with statistics
        """
        base_stats = super().get_statistics()
        
        total_decisions = self.veto_count + self.approval_count
        veto_rate = self.veto_count / total_decisions if total_decisions > 0 else 0.0
        
        dual_process_stats = {
            'type': 'DualProcessOrganism',
            'veto_count': self.veto_count,
            'approval_count': self.approval_count,
            'fallback_count': self.fallback_count,
            'veto_rate': veto_rate,
            'survival_snn_params': self.survival_snn.get_statistics(),
            'ethical_snn_params': self.ethical_snn.get_statistics()
        }
        
        return {**base_stats, **dual_process_stats}
    
    def reset(self) -> None:
        """Reset organism to initial state."""
        # Use stored initial values
        super().reset(position=self.initial_position, initial_energy=self.initial_energy)
        
        # Reset dual-process statistics
        self.veto_count = 0
        self.approval_count = 0
        self.fallback_count = 0
        
        # Reset SNNs (action history tracked by base class)
        # Neural network weights are NOT reset (they persist across episodes)
    
    def __repr__(self) -> str:
        """String representation."""
        veto_rate = self.veto_count / (self.veto_count + self.approval_count) \
                    if (self.veto_count + self.approval_count) > 0 else 0.0
        
        return (
            f"DualProcessOrganism("
            f"id={self.organism_id}, "
            f"pos={self.position}, "
            f"energy={self.energy:.1f}, "
            f"alive={self.alive}, "
            f"vetoes={self.veto_count}, "
            f"veto_rate={veto_rate:.2f})"
        )
