"""
Configuration management using Pydantic.
Loads configuration from YAML files and environment variables.
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "serverless_framework"
    user: str = "serverless_admin"
    password: str = Field(default="", env="DB_PASSWORD")
    pool_size: int = 20
    max_overflow: int = 40
    pool_pre_ping: bool = True
    echo: bool = False
    
    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: str = Field(default="", env="REDIS_PASSWORD")
    db: int = 0
    max_connections: int = 100
    decode_responses: bool = True


class PredictionConfig(BaseSettings):
    """Prediction module configuration"""
    model_type: str = "hybrid"
    model_path: str = "/data/models/prediction/"
    prediction_horizon: int = 600
    update_interval: int = 30
    history_window: int = 86400
    min_data_points: int = 100
    retrain_interval: int = 3600
    retrain_threshold: float = 0.15
    
    # LSTM config
    lstm_input_size: int = 15
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    lstm_batch_size: int = 64
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    
    # Hybrid weights
    arima_weight: float = 0.3
    lstm_weight: float = 0.7


class PlacementConfig(BaseSettings):
    """Placement optimization configuration"""
    algorithm: str = "hungarian"
    optimization_interval: int = 300
    migration_threshold: float = 0.15
    max_concurrent_migrations: int = 5
    migration_cooldown: int = 600
    
    # Cost weights
    user_latency_weight: float = 1.5
    data_locality_weight: float = 1.2
    inter_function_weight: float = 1.0
    resource_utilization_weight: float = 0.8
    migration_cost_weight: float = 2.0


class WarmingConfig(BaseSettings):
    """Container warming configuration"""
    min_warm_pool: int = 2
    max_warm_pool: int = 100
    default_warm_pool: int = 5
    prewarm_buffer: float = 1.2
    prewarm_lead_time: int = 60
    
    # Keep-alive
    adaptive_keepalive: bool = True
    min_keepalive_duration: int = 5
    max_keepalive_duration: int = 600
    default_keepalive_duration: int = 60
    cost_threshold: float = 0.001
    
    pool_check_interval: int = 10
    idle_timeout: int = 300
    drain_timeout: int = 30


class CheckpointConfig(BaseSettings):
    """Checkpoint/restore configuration"""
    enabled: bool = True
    criu_path: str = "/usr/sbin/criu"
    images_dir: str = "/var/lib/serverless/checkpoints"
    checkpoint_after_init: bool = True
    max_checkpoints_per_function: int = 5
    restore_timeout: int = 5
    
    # Compression
    compression_enabled: bool = True
    compression_algorithm: str = "lz4"
    compression_level: int = 3


class CostConfig(BaseSettings):
    """Cost optimization configuration"""
    compute_cost_per_gb_second: float = 0.0000166667
    memory_cost_per_gb_second: float = 0.0000016667
    network_cost_per_gb: float = 0.12
    storage_cost_per_gb_month: float = 0.023
    cold_start_business_cost: float = 0.002
    
    target_cost_reduction: float = 0.30
    max_acceptable_latency: int = 200
    daily_budget: float = 1000.0
    alert_threshold: float = 0.90


class MonitoringConfig(BaseSettings):
    """Monitoring configuration"""
    metrics_interval: int = 10
    metrics_retention: int = 604800
    ebpf_enabled: bool = True
    prometheus_port: int = 9090
    tracing_enabled: bool = True
    tracing_sampling_rate: float = 0.1


class APIConfig(BaseSettings):
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 1000
    
    # Auth
    auth_enabled: bool = True
    jwt_secret: str = Field(default="", env="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    access_token_expire: int = 3600


class Settings(BaseSettings):
    """Main settings class"""
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    prediction: PredictionConfig = PredictionConfig()
    placement: PlacementConfig = PlacementConfig()
    warming: WarmingConfig = WarmingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    cost: CostConfig = CostConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    api: APIConfig = APIConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    config_file = os.getenv("CONFIG_FILE", "config/production.yaml")
    
    # Load YAML configuration
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dict to Settings object
        settings = Settings(**config_dict)
    else:
        # Use defaults if config file doesn't exist
        settings = Settings()
    
    return settings


def reload_settings():
    """Force reload of settings (clear cache)"""
    get_settings.cache_clear()
