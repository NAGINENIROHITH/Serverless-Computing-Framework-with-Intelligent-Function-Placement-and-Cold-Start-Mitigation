"""
Proactive container pre-warming controller.
Uses ML predictions to maintain optimal warm pools.
"""

import asyncio
from typing import Dict, List
from loguru import logger
from src.common.models import Function, Node, WarmPool
from src.common.config import get_settings


from src.warming.keep_alive import AdaptiveKeepAlive

class WarmingController:
    """Manages container warm pools proactively"""
    
    def __init__(self):
        self.config = get_settings().warming
        self.warm_pools: Dict[int, WarmPool] = {}
        self.keep_alive = AdaptiveKeepAlive()
        
    async def start(self):
        """Start warming control loop"""
        logger.info("Starting warming controller")
        
        while True:
            await asyncio.sleep(self.config.pool_check_interval)
            await self._reconcile_warm_pools()
    
    async def _reconcile_warm_pools(self):
        """Ensure warm pools match target sizes"""
        for function_id, pool in self.warm_pools.items():
            if pool.current_size < pool.target_size:
                await self._scale_up(pool, pool.target_size - pool.current_size)
            elif pool.current_size > pool.target_size:
                # Check keep-alive policy before scaling down
                extra_containers = pool.current_size - pool.target_size
                await self._scale_down(pool, extra_containers, str(function_id))
    
    async def _scale_up(self, pool: WarmPool, count: int):
        """Add containers to warm pool"""
        logger.info(f"Scaling up warm pool by {count} containers")
        # Implementation would call Kubernetes/Docker API
        pool.current_size += count
    
    async def _scale_down(self, pool: WarmPool, count: int, function_id: str):
        """Remove containers from warm pool if policy allows"""
        # In a real implementation we would track idle time per container
        # Here we simulate average idle time
        simulated_idle_time = 60.0  
        
        if self.keep_alive.should_terminate(function_id, simulated_idle_time):
            logger.info(f"Scaling down warm pool by {count} containers (TTL expired)")
            pool.current_size = max(0, pool.current_size - count)
        else:
            logger.debug(f"Skipping scale down for {function_id}: Keep-Alive active")
    
    def update_target_size(self, function_id: int, predicted_load: float):
        """Update target pool size based on prediction"""
        avg_exec_time = 0.5  # seconds
        target = int(predicted_load * avg_exec_time * self.config.prewarm_buffer)
        target = max(self.config.min_warm_pool, min(target, self.config.max_warm_pool))
        
        if function_id in self.warm_pools:
            self.warm_pools[function_id].target_size = target
