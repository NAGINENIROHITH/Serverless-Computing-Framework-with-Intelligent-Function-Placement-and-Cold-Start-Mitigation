"""
FastAPI application for Intelligent Serverless Framework.
Serves dynamic simulated data for the dashboard.
"""

import asyncio
import random
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

app = FastAPI(
    title="Intelligent Serverless Framework",
    description="ML-driven serverless platform with predictive optimization",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simulation State ---
class SimulationState:
    def __init__(self):
        self.system_metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": 125847,
            "cold_starts": 10068,
            "warm_starts": 115779,
            "cold_start_percentage": 8.0,
            "p50_latency_ms": 45,
            "p95_latency_ms": 125,
            "p99_latency_ms": 185,
            "mean_latency_ms": 68,
            "avg_cpu_utilization": 42.5,
            "avg_memory_utilization": 38.2,
            "hourly_cost": 4.67,
            "sla_violations": 234,
            "sla_compliance_percentage": 98.5,
            # --- New Advanced Metrics ---
            "cache_hit_rate_l1": 65.4,
            "cache_hit_rate_l2": 24.1,
            "cache_hit_rate_l3": 10.5,
            "federated_accuracy": 89.2,
            "federated_clients_active": 15,
            "keep_alive_savings": 450,
            "federated_clients_active": 15,
            "keep_alive_savings": 450,
            "composition_latency_reduction_ms": 12,
            # --- Dynamic Placement Metrics ---
            "migrations_24h": 23,
            "data_locality_score": 0.92
        }
        
        self.functions = [
            {"id": 1, "name": "image-processor", "status": "active", "total_invocations": 45623, "avg_execution_time_ms": 234},
            {"id": 2, "name": "data-transformer", "status": "active", "total_invocations": 32156, "avg_execution_time_ms": 156},
            {"id": 3, "name": "api-gateway", "status": "active", "total_invocations": 87432, "avg_execution_time_ms": 45},
            {"id": 4, "name": "ml-inference", "status": "idle", "total_invocations": 15678, "avg_execution_time_ms": 567},
            {"id": 5, "name": "notification-service", "status": "active", "total_invocations": 23456, "avg_execution_time_ms": 89}
        ]
        
        self.predictions = [
            {"function_name": "image-processor", "total_predicted": 5234, "model_type": "hybrid"},
            {"function_name": "data-transformer", "total_predicted": 3456, "model_type": "hybrid"},
            {"function_name": "api-gateway", "total_predicted": 8765, "model_type": "hybrid"}
        ]
        
        self.cost_data = {
            "total_cost": 112.45,
            "cost_breakdown": {
                "compute": 45.67,
                "memory": 23.45,
                "network": 15.32,
                "storage": 8.91,
                "cold_start_impact": 19.10
            },
            "invocations": 125847,
            "cost_per_invocation": 0.000893,
            "cost_savings": 42.35,
            "cost_savings_percentage": 27.3
        }
        self.placements = [
            {"function": "image-processor", "node": "edge-node-1", "type": "Edge", "latency": 12},
            {"function": "api-gateway", "node": "cloud-node-us-east", "type": "Cloud", "latency": 45}
        ]

    def update(self):
        """Simulate traffic and metric updates"""
        # Simulate Request Traffic
        # Update basic metrics
        new_requests = random.randint(5, 20)
        self.system_metrics["total_requests"] += new_requests
        
        # Fluctuate Latency
        self.system_metrics["mean_latency_ms"] = max(10, 68 + random.randint(-5, 5))
        self.system_metrics["p50_latency_ms"] = max(10, 45 + random.randint(-3, 3))
        self.system_metrics["p95_latency_ms"] = max(50, 125 + random.randint(-7, 7))
        self.system_metrics["p99_latency_ms"] = max(50, 185 + random.randint(-10, 10))
        
        # --- Simulate Advanced Feature Metrics ---
        # Vary Cache Hits
        self.system_metrics["cache_hit_rate_l1"] = round(min(99.9, max(50.0, self.system_metrics["cache_hit_rate_l1"] + random.uniform(-2, 2))), 1)
        self.system_metrics["cache_hit_rate_l2"] = round(min(40.0, max(10.0, self.system_metrics["cache_hit_rate_l2"] + random.uniform(-1, 1))), 1)
        self.system_metrics["cache_hit_rate_l3"] = round(min(20.0, max(5.0, self.system_metrics["cache_hit_rate_l3"] + random.uniform(-0.5, 0.5))), 1)
        
        # Vary Federated Learning (Slowly improving)
        self.system_metrics["federated_accuracy"] = round(min(99.9, self.system_metrics.get("federated_accuracy", 89.0) + random.uniform(0, 0.05)), 1)
        self.system_metrics["federated_clients_active"] = max(5, min(20, self.system_metrics["federated_clients_active"] + random.randint(-1, 1)))
        
        # Vary Savings
        self.system_metrics["keep_alive_savings"] += random.randint(0, 1)
        self.system_metrics["composition_latency_reduction_ms"] = max(0, 12 + random.randint(-1, 1))
        
        # Fluctuate Cold Starts
        if random.random() > 0.7:
             self.system_metrics["cold_starts"] += random.randint(1, 5)
        
        self.system_metrics["cold_start_percentage"] = round(
            (self.system_metrics["cold_starts"] / self.system_metrics["total_requests"]) * 100, 2
        )
        
        # Fluctuate Placement Metrics
        # Random walk for data locality (more volatile for demo)
        self.system_metrics["data_locality_score"] = round(max(0.6, min(0.99, self.system_metrics["data_locality_score"] + random.uniform(-0.08, 0.08))), 2)
        # Randomly increment migrations (50% chance)
        if random.random() > 0.5:
            self.system_metrics["migrations_24h"] += 1
        
        # Update Functions
        for func in self.functions:
            if func["status"] == "active":
                func["total_invocations"] += random.randint(0, 50)
                # Randomly fluctuate execution time
                func["avg_execution_time_ms"] = max(10, func["avg_execution_time_ms"] + random.randint(-5, 5))

        # Update Predictions occasionally
        for pred in self.predictions:
            pred["total_predicted"] = int(pred["total_predicted"] * (1 + random.uniform(-0.01, 0.01)))

        # Update Cost Data (Simulated)
        # Cost increases as requests increase
        additional_cost = new_requests * self.cost_data["cost_per_invocation"]
        self.cost_data["total_cost"] += additional_cost
        self.cost_data["invocations"] += new_requests
        
        # Fluctuate breakdown slightly
        self.cost_data["cost_breakdown"]["compute"] += additional_cost * 0.4
        self.cost_data["cost_breakdown"]["memory"] += additional_cost * 0.2
        self.cost_data["cost_breakdown"]["network"] += additional_cost * 0.15
        
        # Recalculate savings (mock logic)
        self.cost_data["cost_savings"] = self.cost_data["total_cost"] * (self.cost_data["cost_savings_percentage"] / 100)

        self.system_metrics["timestamp"] = datetime.now().isoformat()

state = SimulationState()

async def run_simulation():
    """Background task to run simulation loop"""
    while True:
        state.update()
        await asyncio.sleep(2)  # Update every 2 seconds

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Intelligent Serverless Framework API", "status": "simulating"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/metrics/system")
async def get_system_metrics():
    return state.system_metrics

@app.get("/api/v1/placements")
async def get_placements():
    """Get current function placements"""
    return {
        "placements": state.placements,
        "optimization_metrics": {
            "avg_latency": state.system_metrics.get("mean_latency_ms", 45),
            "data_locality_score": state.system_metrics.get("data_locality_score", 0.92),
            "migrations_24h": state.system_metrics.get("migrations_24h", 23)
        }
    }

@app.get("/api/v1/functions")
async def list_functions():
    return {"functions": state.functions, "count": len(state.functions)}

@app.post("/api/v1/functions")
async def add_function(bg_tasks: BackgroundTasks): # Mock add
    new_id = len(state.functions) + 1
    new_func = {
        "id": new_id,
        "name": f"new-function-{new_id}",
        "status": "idle",
        "total_invocations": 0,
        "avg_execution_time_ms": 0
    }
    state.functions.append(new_func)
    return {"message": "Function added", "function": new_func}

@app.get("/api/v1/predictions")
async def get_predictions():
    return {"predictions": state.predictions}

@app.get("/api/v1/metrics/cost/current")
async def get_cost():
    return state.cost_data

@app.on_event("startup")
async def startup():
    logger.info("Starting Simulation Loop")
    asyncio.create_task(run_simulation())

@app.on_event("shutdown")
async def shutdown():
    logger.info("Stopping Simulation")

