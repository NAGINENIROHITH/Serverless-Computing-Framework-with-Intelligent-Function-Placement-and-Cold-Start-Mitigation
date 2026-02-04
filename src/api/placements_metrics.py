"""
API endpoints for placement information and system metrics.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from src.common.database import get_db_session
from src.common.models import (
    FunctionPlacement, Node, Function, MetricsSnapshot, CostRecord
)


# Placement Router
placements_router = APIRouter()


class PlacementInfo(BaseModel):
    placement_id: int
    function_id: int
    function_name: str
    node_id: int
    node_name: str
    node_tier: str
    placed_at: datetime
    is_active: bool
    total_cost: float
    user_latency_cost: float
    data_locality_cost: float
    inter_function_cost: float


class NodeInfo(BaseModel):
    node_id: int
    node_name: str
    tier: str
    region: str
    latitude: float
    longitude: float
    cpu_utilization: float
    memory_utilization: float
    is_active: bool
    functions_count: int


@placements_router.get("/", response_model=List[PlacementInfo])
async def get_placements(
    function_id: Optional[int] = None,
    node_id: Optional[int] = None,
    is_active: bool = True,
    session: AsyncSession = Depends(get_db_session),
):
    """Get current function placements."""
    query = select(FunctionPlacement).where(
        FunctionPlacement.is_active == is_active
    )
    
    if function_id:
        query = query.where(FunctionPlacement.function_id == function_id)
    if node_id:
        query = query.where(FunctionPlacement.node_id == node_id)
    
    result = await session.execute(query)
    placements = result.scalars().all()
    
    # Get function and node names
    function_ids = list(set(p.function_id for p in placements))
    node_ids = list(set(p.node_id for p in placements))
    
    func_result = await session.execute(
        select(Function).where(Function.id.in_(function_ids))
    )
    functions = {f.id: f for f in func_result.scalars().all()}
    
    node_result = await session.execute(
        select(Node).where(Node.id.in_(node_ids))
    )
    nodes = {n.id: n for n in node_result.scalars().all()}
    
    return [
        PlacementInfo(
            placement_id=p.id,
            function_id=p.function_id,
            function_name=functions[p.function_id].name,
            node_id=p.node_id,
            node_name=nodes[p.node_id].name,
            node_tier=nodes[p.node_id].tier.value,
            placed_at=p.placed_at,
            is_active=p.is_active,
            total_cost=p.total_cost,
            user_latency_cost=p.user_latency_cost,
            data_locality_cost=p.data_locality_cost,
            inter_function_cost=p.inter_function_cost,
        )
        for p in placements
        if p.function_id in functions and p.node_id in nodes
    ]


@placements_router.get("/nodes", response_model=List[NodeInfo])
async def get_nodes(
    tier: Optional[str] = None,
    is_active: bool = True,
    session: AsyncSession = Depends(get_db_session),
):
    """Get information about infrastructure nodes."""
    query = select(Node).where(Node.is_active == is_active)
    
    if tier:
        query = query.where(Node.tier == tier)
    
    result = await session.execute(query)
    nodes = result.scalars().all()
    
    # Count functions per node
    node_ids = [n.id for n in nodes]
    placement_counts = await session.execute(
        select(
            FunctionPlacement.node_id,
            func.count(FunctionPlacement.id).label('count')
        ).where(
            and_(
                FunctionPlacement.node_id.in_(node_ids),
                FunctionPlacement.is_active == True
            )
        ).group_by(FunctionPlacement.node_id)
    )
    counts = {row.node_id: row.count for row in placement_counts}
    
    return [
        NodeInfo(
            node_id=n.id,
            node_name=n.name,
            tier=n.tier.value,
            region=n.region,
            latitude=n.latitude,
            longitude=n.longitude,
            cpu_utilization=n.cpu_utilization,
            memory_utilization=n.memory_utilization,
            is_active=n.is_active,
            functions_count=counts.get(n.id, 0),
        )
        for n in nodes
    ]


@placements_router.get("/optimization-history")
async def get_optimization_history(
    hours: int = Query(24, ge=1, le=168),
    session: AsyncSession = Depends(get_db_session),
):
    """Get placement optimization history."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    result = await session.execute(
        select(FunctionPlacement).where(
            FunctionPlacement.placed_at >= cutoff
        ).order_by(FunctionPlacement.placed_at.desc())
    )
    placements = result.scalars().all()
    
    # Count migrations
    migrations = [p for p in placements if p.previous_node_id is not None]
    
    # Average costs
    total_cost = sum(p.total_cost for p in placements) / len(placements) if placements else 0
    
    return {
        "time_period": f"Last {hours} hours",
        "total_placements": len(placements),
        "total_migrations": len(migrations),
        "avg_placement_cost": round(total_cost, 2),
        "migration_reasons": {
            p.migration_reason: sum(1 for m in migrations if m.migration_reason == p.migration_reason)
            for p in migrations
            if p.migration_reason
        },
    }


# Metrics Router
metrics_router = APIRouter()


class SystemMetrics(BaseModel):
    timestamp: datetime
    total_requests: int
    cold_starts: int
    warm_starts: int
    cold_start_percentage: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    avg_cpu_utilization: float
    avg_memory_utilization: float
    hourly_cost: float
    sla_violations: int
    sla_compliance_percentage: float


@metrics_router.get("/system", response_model=SystemMetrics)
async def get_system_metrics(
    session: AsyncSession = Depends(get_db_session),
):
    """Get current system-wide metrics."""
    # Get latest snapshot
    result = await session.execute(
        select(MetricsSnapshot).order_by(
            MetricsSnapshot.timestamp.desc()
        ).limit(1)
    )
    snapshot = result.scalar_one_or_none()
    
    if not snapshot:
        raise HTTPException(
            status_code=404,
            detail="No metrics snapshots available"
        )
    
    return SystemMetrics(
        timestamp=snapshot.timestamp,
        total_requests=snapshot.total_requests,
        cold_starts=snapshot.cold_starts,
        warm_starts=snapshot.warm_starts,
        cold_start_percentage=(
            snapshot.cold_starts / snapshot.total_requests * 100
            if snapshot.total_requests > 0 else 0
        ),
        p50_latency_ms=snapshot.p50_latency_ms or 0,
        p95_latency_ms=snapshot.p95_latency_ms or 0,
        p99_latency_ms=snapshot.p99_latency_ms or 0,
        mean_latency_ms=snapshot.mean_latency_ms or 0,
        avg_cpu_utilization=snapshot.avg_cpu_utilization or 0,
        avg_memory_utilization=snapshot.avg_memory_utilization or 0,
        hourly_cost=snapshot.hourly_cost or 0,
        sla_violations=snapshot.sla_violations or 0,
        sla_compliance_percentage=snapshot.sla_compliance_percentage or 0,
    )


@metrics_router.get("/system/history")
async def get_metrics_history(
    hours: int = Query(24, ge=1, le=168),
    session: AsyncSession = Depends(get_db_session),
):
    """Get historical system metrics."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    result = await session.execute(
        select(MetricsSnapshot).where(
            MetricsSnapshot.timestamp >= cutoff
        ).order_by(MetricsSnapshot.timestamp)
    )
    snapshots = result.scalars().all()
    
    return {
        "time_period": f"Last {hours} hours",
        "data_points": len(snapshots),
        "metrics": [
            {
                "timestamp": s.timestamp.isoformat(),
                "cold_start_percentage": (
                    s.cold_starts / s.total_requests * 100
                    if s.total_requests > 0 else 0
                ),
                "p99_latency_ms": s.p99_latency_ms,
                "sla_compliance_percentage": s.sla_compliance_percentage,
                "hourly_cost": s.hourly_cost,
            }
            for s in snapshots
        ]
    }


@metrics_router.get("/cost/current")
async def get_current_cost(
    session: AsyncSession = Depends(get_db_session),
):
    """Get current cost information."""
    # Get today's cost record
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    result = await session.execute(
        select(CostRecord).where(
            CostRecord.period_start >= today_start
        ).order_by(CostRecord.period_start.desc()).limit(1)
    )
    cost_record = result.scalar_one_or_none()
    
    if not cost_record:
        return {
            "period": "today",
            "total_cost": 0.0,
            "cost_breakdown": {},
            "invocations": 0,
            "cost_per_invocation": 0.0,
        }
    
    return {
        "period": f"{cost_record.period_start.date()} to {cost_record.period_end.date()}",
        "total_cost": cost_record.total_cost,
        "cost_breakdown": {
            "compute": cost_record.compute_cost,
            "memory": cost_record.memory_cost,
            "network": cost_record.network_cost,
            "storage": cost_record.storage_cost,
            "cold_start_impact": cost_record.cold_start_business_cost,
        },
        "invocations": cost_record.total_invocations,
        "cost_per_invocation": (
            cost_record.total_cost / cost_record.total_invocations
            if cost_record.total_invocations > 0 else 0
        ),
        "cost_savings": cost_record.cost_savings,
        "cost_savings_percentage": cost_record.cost_savings_percentage,
    }


@metrics_router.get("/cost/history")
async def get_cost_history(
    days: int = Query(7, ge=1, le=90),
    session: AsyncSession = Depends(get_db_session),
):
    """Get cost history over time."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    result = await session.execute(
        select(CostRecord).where(
            CostRecord.period_start >= cutoff
        ).order_by(CostRecord.period_start)
    )
    records = result.scalars().all()
    
    total_cost = sum(r.total_cost for r in records)
    total_savings = sum(r.cost_savings or 0 for r in records)
    
    return {
        "time_period": f"Last {days} days",
        "total_cost": round(total_cost, 2),
        "total_savings": round(total_savings, 2),
        "daily_average": round(total_cost / days, 2) if days > 0 else 0,
        "cost_trend": [
            {
                "date": r.period_start.date().isoformat(),
                "cost": r.total_cost,
                "invocations": r.total_invocations,
                "cost_per_invocation": (
                    r.total_cost / r.total_invocations
                    if r.total_invocations > 0 else 0
                ),
            }
            for r in records
        ]
    }


@metrics_router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "intelligent-serverless-framework",
    }


# Export routers
placements_router_export = placements_router
metrics_router_export = metrics_router
