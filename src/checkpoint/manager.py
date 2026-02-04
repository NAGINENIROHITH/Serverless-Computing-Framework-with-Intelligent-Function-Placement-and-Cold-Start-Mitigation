"""
CRIU-based checkpoint/restore manager for fast container initialization.
"""

import subprocess
import os
from pathlib import Path
from loguru import logger
from src.common.config import get_settings


from src.cache.policy import CachePolicy

class CheckpointManager:
    """Manages container checkpoints using CRIU"""
    
    def __init__(self):
        self.config = get_settings().checkpoint
        self.images_dir = Path(self.config.images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.cache_policy = CachePolicy()
        
    def create_checkpoint(
        self,
        container_id: str,
        function_name: str,
    ) -> str:
        """Create checkpoint of running container"""
        
        checkpoint_path = self.images_dir / f"{function_name}_{container_id}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.config.criu_path, "dump",
            "-t", container_id,
            "--images-dir", str(checkpoint_path),
            "--shell-job",
            "--leave-running"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=self.config.restore_timeout)
            
            if result.returncode == 0:
                logger.info(f"Checkpoint created: {checkpoint_path}")
                # Register with Cache Policy
                self.cache_policy.register_checkpoint(function_name, str(checkpoint_path))
                return str(checkpoint_path)
            else:
                logger.error(f"Checkpoint failed: {result.stderr.decode()}")
                return None
        
        except Exception as e:
            logger.error(f"Checkpoint error: {e}")
            return None
    
    def restore_from_checkpoint(self, checkpoint_path: str, function_id: str = None) -> str:
        """Restore container from checkpoint"""
        
        # Consult Cache Policy if function_id is provided
        if function_id:
            cached_path = self.cache_policy.get_restore_source(function_id)
            if cached_path:
                checkpoint_path = cached_path
        
        cmd = [
            self.config.criu_path, "restore",
            "--images-dir", checkpoint_path,
            "--shell-job"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=self.config.restore_timeout)
            
            if result.returncode == 0:
                logger.info(f"Restored from checkpoint: {checkpoint_path}")
                return "container_id"
            else:
                logger.error(f"Restore failed: {result.stderr.decode()}")
                return None
        
        except Exception as e:
            logger.error(f"Restore error: {e}")
            return None
