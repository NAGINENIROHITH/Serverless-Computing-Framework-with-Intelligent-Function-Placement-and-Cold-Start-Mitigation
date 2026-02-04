"""
Adaptive Keep-Alive Policy for Container Pre-warming.
dynamically adjusts container time-to-live based on invocation patterns.
"""

import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from src.common.config import get_settings

class AdaptiveKeepAlive:
    """
    Manages dynamic keep-alive windows for functions.
    Uses statistical analysis of inter-arrival times to determine optimal TTL.
    """
    
    def __init__(self):
        self.config = get_settings().warming
        self.stats: Dict[str, List[float]] = {}  # History of inter-arrival times
        self.last_invocation: Dict[str, float] = {}
        self.current_ttls: Dict[str, float] = {}
        
    def record_invocation(self, function_id: str, timestamp: float):
        """Record a new invocation timestamp for a function"""
        if function_id in self.last_invocation:
            inter_arrival = timestamp - self.last_invocation[function_id]
            
            if function_id not in self.stats:
                self.stats[function_id] = []
            
            # Keep rolling window of last 100 invocations
            self.stats[function_id].append(inter_arrival)
            if len(self.stats[function_id]) > 100:
                self.stats[function_id].pop(0)
                
            # Update TTL based on new data
            self._update_ttl(function_id)
            
        self.last_invocation[function_id] = timestamp
        
    def _update_ttl(self, function_id: str):
        """Recalculate TTL using Mean + 2*StdDev (95% confidence coverage)"""
        if not self.stats.get(function_id):
            return
            
        data = np.array(self.stats[function_id])
        mean_ia = np.mean(data)
        std_ia = np.std(data)
        
        # Formula: Cover 95% of expected next arrivals
        # Add a safety buffer multiplier from config
        optimal_ttl = mean_ia + (2 * std_ia)
        
        # Clamp to configured limits
        optimal_ttl = max(self.config.min_keep_alive, min(optimal_ttl, self.config.max_keep_alive))
        
        self.current_ttls[function_id] = optimal_ttl
        logger.debug(f"Updated TTL for {function_id}: {optimal_ttl:.2f}s (Mean: {mean_ia:.2f}, Std: {std_ia:.2f})")
        
    def get_ttl(self, function_id: str) -> float:
        """Get current optimal TTL for a function"""
        return self.current_ttls.get(function_id, self.config.default_keep_alive)
        
    def should_terminate(self, function_id: str, idle_time: float) -> bool:
        """Decide if a container should be terminated based on idle time and policy"""
        ttl = self.get_ttl(function_id)
        if idle_time > ttl:
            logger.info(f"Terminating container for {function_id}: Idle {idle_time:.2f}s > TTL {ttl:.2f}s")
            return True
        return False
