"""
Federated Learning Server.
Aggregates model weights from clients using FedAvg algorithm.
"""

import numpy as np
from typing import List, Dict
from loguru import logger

class FederatedServer:
    """
    Central coordinator for Federated Learning.
    Aggregates weights without accessing raw data.
    """
    
    def __init__(self):
        self.global_weights = None
        self.round_number = 0
        
    def aggregate(self, client_updates: List[Dict]) -> List[float]:
        """
        Perform FedAvg aggregation.
        Ref: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data"
        """
        if not client_updates:
            return []
            
        self.round_number += 1
        logger.info(f"Starting FL Round {self.round_number} with {len(client_updates)} clients")
        
        total_samples = sum(u['num_samples'] for u in client_updates)
        weighted_weights = []
        
        # Calculate weighted average of weights
        for update in client_updates:
            weight_vector = np.array(update['weights'])
            sample_weight = update['num_samples'] / total_samples
            weighted_weights.append(weight_vector * sample_weight)
            
        # Sum up weighted vectors
        try:
            new_global_weights = np.sum(weighted_weights, axis=0)
            self.global_weights = new_global_weights.tolist()
            
            logger.success(f"FL Round {self.round_number} complete. Global params updated.")
            return self.global_weights
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return []
