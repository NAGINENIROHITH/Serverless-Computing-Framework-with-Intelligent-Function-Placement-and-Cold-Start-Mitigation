"""
Federated Learning Client.
Simulates local model training and weight updates.
"""

import numpy as np
from typing import List, Dict
from loguru import logger
from src.prediction.arima_model import ARIMAPredictor

class FederatedClient:
    """
    Represents a tenant/node participating in federated learning.
    Trains on local data but shares only model weights.
    """
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.local_model = ARIMAPredictor(auto_tune=False)
        self.local_data: List[float] = []
        
    def add_data(self, value: float):
        """Record local invocation data"""
        self.local_data.append(value)
        # Keep window of recent data
        if len(self.local_data) > 500:
            self.local_data.pop(0)
            
    def train(self) -> Dict:
        """
        Train local model and return weights (parameters).
        Does NOT share raw data.
        """
        if len(self.local_data) < 50:
            logger.warning(f"Client {self.client_id}: Not enough data to train")
            return {}
            
        try:
            logger.info(f"Client {self.client_id}: Starting local training round")
            self.local_model.fit(np.array(self.local_data))
            
            # Extract weights (ARIMA params)
            if self.local_model.fitted_model:
                params = self.local_model.fitted_model.params
                return {
                    'client_id': self.client_id,
                    'num_samples': len(self.local_data),
                    'weights': params.tolist()
                }
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            
        return {}
        
    def update_model(self, global_weights: List[float]):
        """Receive updated global model weights"""
        logger.info(f"Client {self.client_id}: Received global model update")
        # In a real implementation we would set these weights into the model
        pass
