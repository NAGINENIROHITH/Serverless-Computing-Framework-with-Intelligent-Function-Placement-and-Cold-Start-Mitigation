"""
Function placement optimizer using multi-objective optimization.
Considers user latency, data locality, and resource utilization.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
from loguru import logger
from src.common.models import Function, Node
from src.common.config import get_settings


from src.composition.optimizer import CompositionOptimizer

class PlacementOptimizer:
    """Intelligent function placement across edge-cloud continuum"""
    
    def __init__(self):
        self.config = get_settings().placement
        self.cost_matrix = None
        self.composition_opt = CompositionOptimizer()
        
    def calculate_placement_cost(
        self,
        function: Function,
        node: Node,
        user_locations: List[Tuple[float, float]],
    ) -> float:
        """Calculate total placement cost"""
        
        # 1. User latency cost
        user_latency_cost = self._calculate_user_latency(node, user_locations)
        
        # 2. Data locality cost
        data_locality_cost = self._calculate_data_locality(function, node)
        
        # 3. Inter-function communication cost (Graph-aware)
        # We check affinity with other functions ALREADY placed on this node
        inter_func_cost = self._calculate_inter_function_cost(function, node)
        
        # 4. Resource utilization cost
        resource_cost = self._calculate_resource_cost(node)
        
        # Weighted sum
        total_cost = (
            self.config.user_latency_weight * user_latency_cost +
            self.config.data_locality_weight * data_locality_cost +
            self.config.inter_function_weight * inter_func_cost +
            self.config.resource_utilization_weight * resource_cost
        )
        
        return total_cost
    
    def optimize(
        self,
        functions: List[Function],
        nodes: List[Node],
    ) -> Dict[int, int]:
        """Find optimal function-to-node placement"""
        
        # Build cost matrix
        n_functions = len(functions)
        n_nodes = len(nodes)
        
        cost_matrix = np.zeros((n_functions, n_nodes))
        
        for i, func in enumerate(functions):
            for j, node in enumerate(nodes):
                cost_matrix[i, j] = self.calculate_placement_cost(
                    func, node, []
                )
        
        # Hungarian algorithm for optimal assignment
        func_indices, node_indices = linear_sum_assignment(cost_matrix)
        
        # Create placement mapping
        placement = {
            functions[i].id: nodes[j].id
            for i, j in zip(func_indices, node_indices)
        }
        
        return placement
    
    def _calculate_user_latency(
        self,
        node: Node,
        user_locations: List[Tuple[float, float]]
    ) -> float:
        """Calculate average latency to users"""
        if not user_locations:
            return 0.0
        
        latencies = []
        for lat, lon in user_locations:
            distance = self._haversine_distance(
                (node.latitude, node.longitude),
                (lat, lon)
            )
            latency = distance / 200  # km to ms (fiber optic speed)
            latencies.append(latency)
        
        return np.mean(latencies)
    
    @staticmethod
    def _haversine_distance(coord1, coord2) -> float:
        """Calculate distance between two coordinates in km"""
        from math import radians, sin, cos, sqrt, asin
        
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return 6371 * c  # Earth radius in km
    
    def _calculate_data_locality(self, function: Function, node: Node) -> float:
        """Calculate cost of data access from this node"""
        # Simplified - would query actual data source locations
        return 10.0
    
    def _calculate_inter_function_cost(self, function: Function, node: Node) -> float:
        """Calculate inter-function communication cost"""
        # Lower cost if node hosts high-affinity functions
        affinity_score = 0
        if hasattr(node, 'running_functions'):
             for other_func_id in node.running_functions:
                 affinity_score += self.composition_opt.get_affinity_score(function.id, other_func_id)
        
        # Invert score because we want COST (high affinity = low cost)
        return max(0, 100 - affinity_score)
    
    def _calculate_resource_cost(self, node: Node) -> float:
        """Penalize overloaded nodes"""
        utilization = node.cpu_utilization / 100.0
        if utilization > 0.8:
            return utilization * 100
        return utilization * 10
