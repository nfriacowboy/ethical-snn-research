"""Supervised trainer for ethical SNN."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
from .ethical_dataset import EthicalDataset


class SupervisedTrainer:
    """Supervised pre-trainer for SNN-E.
    
    Trains ethical processing network on synthetic labeled data.
    """
    
    def __init__(self, network: nn.Module, 
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 50):
        """Initialize supervised trainer.
        
        Args:
            network: The ethical SNN to train
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
        """
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Optimizer and loss
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train(self, dataset: EthicalDataset) -> Dict[str, Any]:
        """Train network on ethical dataset.
        
        Args:
            dataset: Ethical dataset for training
            
        Returns:
            Training statistics
        """
        # Split data
        train_scenarios, val_scenarios, train_labels, val_labels = \
            dataset.get_train_test_split(test_ratio=0.2)
        
        print(f"Training on {len(train_scenarios)} scenarios, validating on {len(val_scenarios)}")
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_scenarios, train_labels, dataset)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_scenarios, val_labels, dataset)
            
            # Log
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return self.get_statistics()
    
    def _train_epoch(self, scenarios: List[Dict], labels: List[int], 
                     dataset: EthicalDataset) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            scenarios: Training scenarios
            labels: Training labels
            dataset: Dataset for encoding
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Create batches
        num_batches = len(scenarios) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            
            batch_scenarios = scenarios[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            # Encode scenarios
            batch_features = torch.stack([
                torch.tensor(dataset.encode_scenario(s), dtype=torch.float32)
                for s in batch_scenarios
            ])
            batch_targets = torch.tensor(batch_labels, dtype=torch.long)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.network(batch_features)
            loss = self.criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_targets).sum().item()
            total += batch_targets.size(0)
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, scenarios: List[Dict], labels: List[int],
                  dataset: EthicalDataset) -> Tuple[float, float]:
        """Validate network.
        
        Args:
            scenarios: Validation scenarios
            labels: Validation labels
            dataset: Dataset for encoding
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Encode all scenarios
            features = torch.stack([
                torch.tensor(dataset.encode_scenario(s), dtype=torch.float32)
                for s in scenarios
            ])
            targets = torch.tensor(labels, dtype=torch.long)
            
            # Forward pass
            outputs = self.network(features)
            loss = self.criterion(outputs, targets)
            
            # Statistics
            total_loss = loss.item()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
        
        accuracy = correct / total
        
        return total_loss, accuracy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        return {
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_val_acc': self.val_accuracies[-1] if self.val_accuracies else 0,
            'best_val_acc': max(self.val_accuracies) if self.val_accuracies else 0,
            'num_epochs': self.num_epochs
        }
    
    def save_model(self, path: str):
        """Save trained model.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
    
    def load_model(self, path: str):
        """Load trained model.
        
        Args:
            path: Path to model file
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
