"""
Function Composition Optimizer.
Analyzes function call graphs to identify affinity groups for co-location.
"""

import networkx as nx
from typing import List, Dict, Set, Tuple
from loguru import logger
from src.common.models import Function

class CompositionOptimizer:
    """
    Optimizes function placement based on call chains (DAGs).
    Functions that call each other frequently are grouped into "Affinity Groups".
    """
    
    def __init__(self):
        self.call_graph = nx.DiGraph()
        self.affinity_groups: Dict[int, int] = {}  # function_id -> group_id
        
    def add_dependency(self, caller: str, callee: str, weight: float = 1.0):
        """Register a call dependency between two functions"""
        self.call_graph.add_edge(caller, callee, weight=weight)
        
    def analyze_graph(self):
        """Analyze the call graph to find tightly coupled components"""
        if self.call_graph.number_of_nodes() == 0:
            return

        # Find weakly connected components (clusters of related functions)
        components = list(nx.weakly_connected_components(self.call_graph))
        
        # Assign group IDs
        self.affinity_groups.clear()
        for group_id, component in enumerate(components):
            for func_name in component:
                # In a real system we would map name to ID properly
                # Here we hash the name for a simple ID simulation
                func_id = abs(hash(func_name)) % 10000 
                self.affinity_groups[func_id] = group_id
                
        logger.info(f"Identified {len(components)} affinity groups in function composition")
        
    def get_affinity_score(self, func1_id: int, func2_id: int) -> float:
        """
        Return a high score if two functions belong to the same affinity group.
        Used by PlacementOptimizer to encourage co-location.
        """
        group1 = self.affinity_groups.get(func1_id)
        group2 = self.affinity_groups.get(func2_id)
        
        if group1 is not None and group1 == group2:
            return 100.0  # High affinity
        return 0.0
        
    def get_critical_path(self) -> List[str]:
        """Identify the longest latency path in the composition"""
        try:
            return nx.dag_longest_path(self.call_graph)
        except Exception:
            return []
