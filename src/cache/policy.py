"""
Tiered Caching Policy.
Manages container image/state caching across Memory (Hot), Disk (Warm), and Remote (Cold) tiers.
"""

from enum import Enum
from typing import Dict, Optional
from loguru import logger
from src.common.config import get_settings

class CacheTier(Enum):
    L1_MEMORY = "hot"     # Running container (Paused)
    L2_DISK = "warm"      # CRIU Checkpoint on disk
    L3_REMOTE = "cold"    # Container Image in Registry

class CachePolicy:
    """
    Determines the best source to restore a function from.
    Prioritizes L1 -> L2 -> L3 to minimize cold start latency.
    """
    
    def __init__(self):
        self.config = get_settings().checkpoint
        # In-memory registry of cached items (simulated)
        self.l1_cache: Dict[str, str] = {}
        self.l2_cache: Dict[str, str] = {}
        
    def register_checkpoint(self, function_id: str, path: str):
        """Register a new disk checkpoint (L2)"""
        self.l2_cache[function_id] = path
        logger.info(f"Registered L2 cache entry for {function_id}")
        
    def get_restore_source(self, function_id: str) -> Optional[str]:
        """
        Find best restore source.
        Returns: Path to checkpoint or None (if cold start needed)
        """
        # Check L1 (Hot) - simulated here as we don't track actual paused containers in this object
        if function_id in self.l1_cache:
            logger.info(f"Cache Hit: L1 (Memory) for {function_id}")
            return self.l1_cache[function_id]
            
        # Check L2 (Warm)
        if function_id in self.l2_cache:
            logger.info(f"Cache Hit: L2 (Disk) for {function_id}")
            return self.l2_cache[function_id]
            
        # Fallback to L3 (Cold)
        logger.info(f"Cache Miss: L3 (Cold Start) required for {function_id}")
        return None
