"""Tests for organism implementations."""

import pytest
import torch
from src.organisms.base_organism import BaseOrganism
from src.organisms.survival_snn import SurvivalSNN
from src.organisms.ethical_snn import EthicalSNN


def test_survival_snn_initialization():
    """Test SurvivalSNN initialization."""
    organism = SurvivalSNN(
        organism_id=0,
        position=(10, 10),
        energy=100.0
    )
    
    assert organism.organism_id == 0
    assert organism.position == (10, 10)
    assert organism.energy == 100.0
    assert organism.alive == True
    assert organism.age == 0


def test_survival_snn_perceive():
    """Test perception functionality."""
    organism = SurvivalSNN(organism_id=0, position=(10, 10))
    
    env_state = {
        'nearest_food': (15, 15),
        'grid': torch.zeros(50, 50)
    }
    
    sensory_input = organism.perceive(env_state)
    
    assert sensory_input.shape[0] == organism.input_size
    assert isinstance(sensory_input, torch.Tensor)


def test_survival_snn_decide():
    """Test decision making."""
    organism = SurvivalSNN(organism_id=0, position=(10, 10))
    
    sensory_input = torch.randn(organism.input_size)
    action = organism.decide(sensory_input)
    
    assert isinstance(action, int)
    assert 0 <= action < organism.output_size


def test_survival_snn_act():
    """Test action execution."""
    organism = SurvivalSNN(organism_id=0, position=(25, 25))
    
    action = 0  # North
    new_position = organism.act(action)
    
    assert isinstance(new_position, tuple)
    assert len(new_position) == 2
    assert organism.position == new_position


def test_ethical_snn_initialization():
    """Test EthicalSNN initialization."""
    organism = EthicalSNN(
        organism_id=0,
        position=(10, 10),
        energy=100.0,
        ethical_weight=0.5
    )
    
    assert organism.organism_id == 0
    assert organism.ethical_weight == 0.5
    assert hasattr(organism, 'ethical_network')


def test_ethical_snn_ethical_context():
    """Test ethical context encoding."""
    organism = EthicalSNN(organism_id=0, position=(10, 10))
    
    env_state = {
        'nearby_organisms': [
            {'organism_id': 1, 'energy': 50.0, 'position': (11, 10)},
            {'organism_id': 2, 'energy': 80.0, 'position': (10, 11)}
        ]
    }
    
    ethical_context = organism.encode_ethical_context(env_state)
    
    assert ethical_context.shape[0] == organism.ethical_input_size
    assert isinstance(ethical_context, torch.Tensor)


def test_organism_update():
    """Test organism state update."""
    organism = SurvivalSNN(organism_id=0, position=(10, 10), energy=50.0)
    
    organism.update(-10.0)
    
    assert organism.energy == 40.0
    assert organism.age == 1
    assert organism.alive == True
    
    organism.update(-50.0)
    
    assert organism.energy == -10.0
    assert organism.alive == False


def test_organism_history_logging():
    """Test history logging."""
    organism = SurvivalSNN(organism_id=0, position=(10, 10))
    
    organism.log_state()
    organism.update(-5.0)
    organism.log_state()
    
    assert len(organism.history) == 2
    assert organism.history[0]['age'] == 0
    assert organism.history[1]['age'] == 1
