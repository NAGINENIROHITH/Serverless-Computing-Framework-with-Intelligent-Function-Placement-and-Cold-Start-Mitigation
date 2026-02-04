"""
Database models for the Intelligent Serverless Framework.
Defines all tables and relationships for persistence layer.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, JSON, Enum, Text, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class FunctionStatus(enum.Enum):
    """Function deployment status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class InvocationStatus(enum.Enum):
    """Invocation execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    COLD_START = "cold_start"


class NodeTier(enum.Enum):
    """Infrastructure tier for placement"""
    EDGE = "edge"
    REGIONAL = "regional"
    CLOUD = "cloud"


class Function(Base):
    """Function metadata and configuration"""
    __tablename__ = "functions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    version = Column(String(50), nullable=False)
    runtime = Column(String(50), nullable=False)  # python3.10, nodejs18, etc.
    
    # Resource requirements
    memory_mb = Column(Integer, default=256)
    cpu_cores = Column(Float, default=0.5)
    timeout_seconds = Column(Integer, default=60)
    
    # Code and dependencies
    code_hash = Column(String(64), nullable=False)
    code_size_bytes = Column(Integer, nullable=False)
    docker_image = Column(String(512), nullable=False)
    
    # Configuration
    environment_vars = Column(JSON, default={})
    layers = Column(JSON, default=[])  # Dependency layers
    
    # Status
    status = Column(Enum(FunctionStatus), default=FunctionStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Statistics
    total_invocations = Column(Integer, default=0)
    avg_execution_time_ms = Column(Float, default=0.0)
    avg_memory_usage_mb = Column(Float, default=0.0)
    
    # Relationships
    invocations = relationship("Invocation", back_populates="function", cascade="all, delete-orphan")
    placements = relationship("FunctionPlacement", back_populates="function", cascade="all, delete-orphan")
    predictions = relationship("InvocationPrediction", back_populates="function", cascade="all, delete-orphan")
    warm_pools = relationship("WarmPool", back_populates="function", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_function_status', 'status'),
        Index('idx_function_runtime', 'runtime'),
    )


class Node(Base):
    """Infrastructure node (edge/regional/cloud)"""
    __tablename__ = "nodes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    tier = Column(Enum(NodeTier), nullable=False)
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    region = Column(String(100), nullable=False)
    availability_zone = Column(String(100))
    
    # Capacity
    total_cpu_cores = Column(Float, nullable=False)
    total_memory_mb = Column(Integer, nullable=False)
    total_storage_gb = Column(Integer, nullable=False)
    
    # Current utilization
    used_cpu_cores = Column(Float, default=0.0)
    used_memory_mb = Column(Integer, default=0)
    used_storage_gb = Column(Integer, default=0)
    
    # Network
    network_bandwidth_gbps = Column(Float, default=1.0)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    placements = relationship("FunctionPlacement", back_populates="node", cascade="all, delete-orphan")
    warm_pools = relationship("WarmPool", back_populates="node", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_node_tier', 'tier'),
        Index('idx_node_active', 'is_active'),
    )
    
    @property
    def cpu_utilization(self) -> float:
        """Calculate CPU utilization percentage"""
        if self.total_cpu_cores == 0:
            return 0.0
        return (self.used_cpu_cores / self.total_cpu_cores) * 100
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage"""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


class Invocation(Base):
    """Individual function invocation record"""
    __tablename__ = "invocations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    function_id = Column(Integer, ForeignKey("functions.id"), nullable=False, index=True)
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_ms = Column(Float)
    
    # Execution details
    is_cold_start = Column(Boolean, default=False, index=True)
    cold_start_duration_ms = Column(Float)
    execution_duration_ms = Column(Float)
    
    # Resource usage
    memory_used_mb = Column(Float)
    cpu_used_cores = Column(Float)
    
    # Context
    node_name = Column(String(255))
    user_location_lat = Column(Float)
    user_location_lon = Column(Float)
    
    # Status
    status = Column(Enum(InvocationStatus), nullable=False)
    error_message = Column(Text)
    
    # Metadata
    request_size_bytes = Column(Integer)
    response_size_bytes = Column(Integer)
    
    # Time features (for ML)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)
    
    # Relationships
    function = relationship("Function", back_populates="invocations")
    
    __table_args__ = (
        Index('idx_invocation_timestamp', 'timestamp'),
        Index('idx_invocation_cold_start', 'is_cold_start'),
        Index('idx_invocation_function_timestamp', 'function_id', 'timestamp'),
    )


class FunctionPlacement(Base):
    """Current and historical function placement"""
    __tablename__ = "function_placements"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    function_id = Column(Integer, ForeignKey("functions.id"), nullable=False, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False, index=True)
    
    # Placement timing
    placed_at = Column(DateTime, default=datetime.utcnow, index=True)
    removed_at = Column(DateTime, index=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # Placement cost/benefit
    user_latency_cost = Column(Float, default=0.0)
    data_locality_cost = Column(Float, default=0.0)
    inter_function_cost = Column(Float, default=0.0)
    resource_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # Migration info
    previous_node_id = Column(Integer, ForeignKey("nodes.id"))
    migration_reason = Column(String(255))
    migration_duration_ms = Column(Float)
    
    # Relationships
    function = relationship("Function", back_populates="placements")
    node = relationship("Node", back_populates="placements", foreign_keys=[node_id])
    
    __table_args__ = (
        Index('idx_placement_active', 'is_active'),
        Index('idx_placement_function_active', 'function_id', 'is_active'),
        UniqueConstraint('function_id', 'node_id', 'placed_at', name='uq_placement'),
    )


class InvocationPrediction(Base):
    """ML predictions for function invocations"""
    __tablename__ = "invocation_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    function_id = Column(Integer, ForeignKey("functions.id"), nullable=False, index=True)
    
    # Prediction details
    predicted_at = Column(DateTime, default=datetime.utcnow, index=True)
    prediction_window_start = Column(DateTime, nullable=False, index=True)
    prediction_window_end = Column(DateTime, nullable=False)
    predicted_invocations = Column(Float, nullable=False)
    
    # Model information
    model_type = Column(String(50), nullable=False)  # lstm, arima, hybrid
    model_version = Column(String(50), nullable=False)
    confidence_score = Column(Float)
    
    # Actual vs predicted (filled after window)
    actual_invocations = Column(Float)
    absolute_error = Column(Float)
    percentage_error = Column(Float)
    
    # Features used
    features = Column(JSON)
    
    # Relationships
    function = relationship("Function", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_window', 'prediction_window_start', 'prediction_window_end'),
        Index('idx_prediction_function_window', 'function_id', 'prediction_window_start'),
    )


class WarmPool(Base):
    """Warm container pool management"""
    __tablename__ = "warm_pools"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    function_id = Column(Integer, ForeignKey("functions.id"), nullable=False, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False, index=True)
    
    # Pool configuration
    target_size = Column(Integer, default=0)
    current_size = Column(Integer, default=0)
    
    # Container details
    container_ids = Column(JSON, default=[])
    
    # Checkpoint info
    checkpoint_path = Column(String(512))
    checkpoint_created_at = Column(DateTime)
    checkpoint_size_bytes = Column(Integer)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_warmed_at = Column(DateTime)
    
    # Keep-alive
    keep_alive_duration_seconds = Column(Integer, default=60)
    idle_since = Column(DateTime)
    
    # Statistics
    total_warm_starts = Column(Integer, default=0)
    avg_warm_start_time_ms = Column(Float, default=0.0)
    
    # Relationships
    function = relationship("Function", back_populates="warm_pools")
    node = relationship("Node", back_populates="warm_pools")
    
    __table_args__ = (
        Index('idx_warm_pool_function_node', 'function_id', 'node_id'),
        UniqueConstraint('function_id', 'node_id', name='uq_warm_pool'),
    )


class CostRecord(Base):
    """Cost tracking and analysis"""
    __tablename__ = "cost_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    
    # Cost breakdown
    compute_cost = Column(Float, default=0.0)
    memory_cost = Column(Float, default=0.0)
    network_cost = Column(Float, default=0.0)
    storage_cost = Column(Float, default=0.0)
    cold_start_business_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # Metrics
    total_invocations = Column(Integer, default=0)
    cold_starts = Column(Integer, default=0)
    warm_starts = Column(Integer, default=0)
    
    # Per-function costs (JSON)
    function_costs = Column(JSON, default={})
    
    # Optimization impact
    cost_without_optimization = Column(Float)
    cost_savings = Column(Float)
    cost_savings_percentage = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_cost_period', 'period_start', 'period_end'),
    )


class FunctionCallGraph(Base):
    """Function composition and call relationships"""
    __tablename__ = "function_call_graphs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Function relationship
    caller_function_id = Column(Integer, ForeignKey("functions.id"), nullable=False, index=True)
    callee_function_id = Column(Integer, ForeignKey("functions.id"), nullable=False, index=True)
    
    # Call statistics
    call_frequency = Column(Integer, default=0)  # calls per hour
    avg_payload_size_bytes = Column(Integer, default=0)
    avg_latency_ms = Column(Float, default=0.0)
    
    # Timing
    first_observed = Column(DateTime, default=datetime.utcnow)
    last_observed = Column(DateTime, default=datetime.utcnow)
    
    # Fusion recommendation
    fusion_recommended = Column(Boolean, default=False)
    fusion_benefit_ms = Column(Float)
    fusion_created_at = Column(DateTime)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_call_graph_caller', 'caller_function_id'),
        Index('idx_call_graph_callee', 'callee_function_id'),
        UniqueConstraint('caller_function_id', 'callee_function_id', name='uq_call_edge'),
    )


class MLModel(Base):
    """ML model versioning and metadata"""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # lstm, arima, hybrid
    
    # Model artifacts
    model_path = Column(String(512), nullable=False)
    config = Column(JSON)
    
    # Training info
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_duration_seconds = Column(Integer)
    training_samples = Column(Integer)
    
    # Performance metrics
    mape = Column(Float)  # Mean Absolute Percentage Error
    rmse = Column(Float)  # Root Mean Squared Error
    mae = Column(Float)   # Mean Absolute Error
    r2_score = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    deployed_at = Column(DateTime)
    deprecated_at = Column(DateTime)
    
    # Federated learning
    is_global_model = Column(Boolean, default=False)
    federated_round = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_name_version', 'name', 'version'),
        Index('idx_model_active', 'is_active'),
        UniqueConstraint('name', 'version', name='uq_model_version'),
    )


class MetricsSnapshot(Base):
    """System-wide metrics snapshots"""
    __tablename__ = "metrics_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Performance metrics
    total_requests = Column(Integer, default=0)
    cold_starts = Column(Integer, default=0)
    warm_starts = Column(Integer, default=0)
    
    # Latency distribution
    p50_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    mean_latency_ms = Column(Float)
    
    # Resource utilization
    avg_cpu_utilization = Column(Float)
    avg_memory_utilization = Column(Float)
    
    # Cost
    hourly_cost = Column(Float)
    
    # Prediction accuracy
    avg_prediction_mape = Column(Float)
    
    # SLA compliance
    sla_violations = Column(Integer, default=0)
    sla_compliance_percentage = Column(Float)
    
    __table_args__ = (
        Index('idx_metrics_timestamp', 'timestamp'),
    )


# Create all tables
def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(engine)


# Drop all tables (for testing)
def drop_tables(engine):
    """Drop all database tables"""
    Base.metadata.drop_all(engine)
