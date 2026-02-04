"""
LSTM-based prediction model for serverless function invocations.
Implements time series forecasting using PyTorch LSTM networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger


class LSTMPredictor(nn.Module):
    """
    LSTM network for time series prediction of function invocations.
    Uses multi-layer LSTM with attention mechanism for better prediction.
    """
    
    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            hidden: Optional hidden state tuple (h, c)
        
        Returns:
            output: Predictions of shape (batch, output_size)
            hidden: Updated hidden state tuple
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep for prediction
        last_hidden = attn_out[:, -1, :]
        
        # Final prediction
        output = self.fc(last_hidden)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for LSTM"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)


class LSTMTrainer:
    """Trainer for LSTM prediction model"""
    
    def __init__(
        self,
        model: LSTMPredictor,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
        )
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
    ) -> dict:
        """
        Full training loop with early stopping.
        
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
        }
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best val loss: {best_val_loss:.6f}")
        return history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")


class LSTMPredictor:
    """High-level interface for LSTM predictions"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        self.model = LSTMPredictor(
            input_size=config.get('input_size', 15),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.2),
        )
        
        self.trainer = LSTMTrainer(
            self.model,
            learning_rate=config.get('learning_rate', 0.001),
            device=self.device,
        )
    
    def predict(
        self,
        input_sequence: np.ndarray,
        steps_ahead: int = 1,
    ) -> np.ndarray:
        """
        Make predictions for future timesteps.
        
        Args:
            input_sequence: Historical data of shape (sequence_length, features)
            steps_ahead: Number of future timesteps to predict
        
        Returns:
            Predictions of shape (steps_ahead,)
        """
        self.model.eval()
        
        # Convert to tensor
        x = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
        
        predictions = []
        hidden = None
        
        with torch.no_grad():
            for _ in range(steps_ahead):
                # Predict next timestep
                output, hidden = self.model(x, hidden)
                prediction = output.cpu().numpy()[0, 0]
                predictions.append(prediction)
                
                # Use prediction as input for next step (autoregressive)
                # Create new input with prediction
                next_input = x[:, -1:, :].clone()
                next_input[0, 0, 0] = prediction  # Update first feature
                x = torch.cat([x[:, 1:, :], next_input], dim=1)
        
        return np.array(predictions)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """Calculate prediction accuracy metrics"""
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R² Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
        }
