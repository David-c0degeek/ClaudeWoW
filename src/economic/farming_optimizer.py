"""
Farming optimization system for resource gathering and route planning.
"""
from typing import Dict, List, Tuple, Set, Optional
import logging
import json
import math
import heapq
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta

from ..utils.config import Config
from .market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ResourceNode:
    """
    Represents a gatherable resource node in the game world.
    """
    node_id: int
    resource_type: str  # herb, ore, leather, etc.
    name: str
    x: float
    y: float
    zone: str
    level_req: int = 1
    skill_req: int = 1
    respawn_time: int = 300  # in seconds
    competition_level: float = 0.0  # 0-1 scale, higher means more competition
    last_visited: Optional[datetime] = None
    estimated_value: int = 0  # Estimated gold value
    
    def distance_to(self, other: 'ResourceNode') -> float:
        """Calculate distance to another node."""
        # Check if in same zone
        if self.zone != other.zone:
            return float('inf')
            
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        
    @property
    def is_available(self) -> bool:
        """Check if node is likely available based on respawn time."""
        if not self.last_visited:
            return True
            
        now = datetime.now()
        return (now - self.last_visited).total_seconds() >= self.respawn_time
        
    @property
    def expected_value(self) -> float:
        """Calculate expected value considering competition."""
        return self.estimated_value * (1 - self.competition_level)

@dataclass
class FarmingRoute:
    """
    Represents an optimized route for farming resources.
    """
    nodes: List[ResourceNode] = field(default_factory=list)
    total_distance: float = 0.0
    total_value: int = 0
    zone: str = ""
    resource_types: Set[str] = field(default_factory=set)
    estimated_time_minutes: float = 0.0
    gold_per_hour: float = 0.0
    
    def add_node(self, node: ResourceNode, distance_from_last: float) -> None:
        """Add a node to the route."""
        self.nodes.append(node)
        self.total_distance += distance_from_last
        self.total_value += node.estimated_value
        self.zone = node.zone  # All nodes should be in same zone
        self.resource_types.add(node.resource_type)
        
        # Update estimated time (assuming 60 units of distance per minute + 20 seconds per node)
        self.estimated_time_minutes = (self.total_distance / 60) + (len(self.nodes) * (20/60))
        
        # Update gold per hour
        if self.estimated_time_minutes > 0:
            self.gold_per_hour = (self.total_value / self.estimated_time_minutes) * 60
            
    @property
    def summary(self) -> Dict:
        """Get summary information about the route."""
        return {
            "zone": self.zone,
            "nodes": len(self.nodes),
            "resource_types": list(self.resource_types),
            "total_value": self.total_value,
            "time_minutes": round(self.estimated_time_minutes, 1),
            "gold_per_hour": round(self.gold_per_hour),
            "total_distance": round(self.total_distance, 1)
        }

class FarmingOptimizer:
    """
    Optimizes resource gathering routes based on efficiency and market value.
    
    This class handles:
    - Resource node mapping and tracking
    - Optimal farming route calculation
    - Time/gold efficiency modeling
    - Dynamic route adjustment based on competition
    """
    
    def __init__(self, config: Config, market_analyzer: MarketAnalyzer):
        """
        Initialize the FarmingOptimizer.
        
        Args:
            config: Application configuration
            market_analyzer: Market analyzer for price data
        """
        self.config = config
        self.market_analyzer = market_analyzer
        self.nodes: Dict[int, ResourceNode] = {}
        self.nodes_by_zone: Dict[str, List[int]] = {}
        self.resource_value_cache: Dict[str, int] = {}
        
        self.nodes_data_file = self.config.get(
            "paths.economic.nodes_data", 
            "data/economic/resource_nodes.json"
        )
        self.load_nodes_data()
        
        # Character movement speed in units per second (default walking speed)
        self.movement_speed = self.config.get("character.movement_speed", 7.0)
        
    def load_nodes_data(self) -> None:
        """Load existing node data from disk."""
        try:
            with open(self.nodes_data_file, 'r') as f:
                data = json.load(f)
                
            for node_data in data:
                node_id = node_data["node_id"]
                node = ResourceNode(
                    node_id=node_id,
                    resource_type=node_data["resource_type"],
                    name=node_data["name"],
                    x=node_data["x"],
                    y=node_data["y"],
                    zone=node_data["zone"],
                    level_req=node_data.get("level_req", 1),
                    skill_req=node_data.get("skill_req", 1),
                    respawn_time=node_data.get("respawn_time", 300),
                    competition_level=node_data.get("competition_level", 0.0),
                    last_visited=datetime.fromisoformat(node_data["last_visited"]) 
                        if "last_visited" in node_data else None,
                    estimated_value=node_data.get("estimated_value", 0)
                )
                
                self.nodes[node_id] = node
                
                # Index by zone
                if node.zone not in self.nodes_by_zone:
                    self.nodes_by_zone[node.zone] = []
                self.nodes_by_zone[node.zone].append(node_id)
                
            logger.info(f"Loaded {len(self.nodes)} resource nodes across {len(self.nodes_by_zone)} zones")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No previous node data found or file is corrupted")
            self.nodes = {}
            self.nodes_by_zone = {}
            
    def save_nodes_data(self) -> None:
        """Save node data to disk."""
        data = []
        
        for node_id, node in self.nodes.items():
            node_data = {
                "node_id": node.node_id,
                "resource_type": node.resource_type,
                "name": node.name,
                "x": node.x,
                "y": node.y,
                "zone": node.zone,
                "level_req": node.level_req,
                "skill_req": node.skill_req,
                "respawn_time": node.respawn_time,
                "competition_level": node.competition_level,
                "estimated_value": node.estimated_value
            }
            
            if node.last_visited:
                node_data["last_visited"] = node.last_visited.isoformat()
                
            data.append(node_data)
            
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.nodes_data_file), exist_ok=True)
            
            with open(self.nodes_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data for {len(self.nodes)} resource nodes")
        except Exception as e:
            logger.error(f"Failed to save node data: {e}")
            
    def add_node(self, node: ResourceNode) -> None:
        """
        Add a new resource node or update an existing one.
        
        Args:
            node: The resource node to add
        """
        self.nodes[node.node_id] = node
        
        # Index by zone
        if node.zone not in self.nodes_by_zone:
            self.nodes_by_zone[node.zone] = []
        if node.node_id not in self.nodes_by_zone[node.zone]:
            self.nodes_by_zone[node.zone].append(node.node_id)
            
        # Update the estimated value based on market data
        self._update_node_value(node)
        
        # Save updated data
        self.save_nodes_data()
        
    def record_node_visit(self, node_id: int) -> None:
        """
        Record that a node has been visited.
        
        Args:
            node_id: ID of the visited node
        """
        if node_id in self.nodes:
            self.nodes[node_id].last_visited = datetime.now()
            self.save_nodes_data()
            
    def update_competition_levels(self, zone: str, competition_data: Dict[int, float]) -> None:
        """
        Update competition levels for nodes in a zone.
        
        Args:
            zone: Zone name
            competition_data: Dictionary mapping node IDs to competition levels (0-1)
        """
        if zone not in self.nodes_by_zone:
            return
            
        updated = False
        for node_id in self.nodes_by_zone[zone]:
            if node_id in competition_data:
                self.nodes[node_id].competition_level = competition_data[node_id]
                updated = True
                
        if updated:
            self.save_nodes_data()
            
    def _update_node_value(self, node: ResourceNode) -> None:
        """
        Update a node's estimated value based on market data.
        
        Args:
            node: Node to update
        """
        # Try to use cached value first
        if node.name in self.resource_value_cache:
            node.estimated_value = self.resource_value_cache[node.name]
            return
            
        # Look up in market analyzer by resource name
        # In a real implementation, we would have a proper mapping from node types to items
        # For now, we'll just use a simple mapping based on node name
        
        # This is a placeholder that would use real market data in the actual implementation
        # Here we're setting some reasonable default values for demonstration
        base_values = {
            "herb": 50,
            "ore": 75,
            "leather": 40,
            "cloth": 35,
            "fish": 60,
            "enchanting": 100,
            "timber": 45
        }
        
        resource_type = node.resource_type.lower()
        if resource_type in base_values:
            base_value = base_values[resource_type]
            
            # Adjust based on skill requirement (higher skill = more valuable)
            skill_multiplier = 1.0 + (node.skill_req / 300)
            
            # Final estimated value
            node.estimated_value = int(base_value * skill_multiplier)
            
            # Cache for future use
            self.resource_value_cache[node.name] = node.estimated_value
            
    def calculate_optimal_route(
        self, 
        zone: str, 
        resource_types: Optional[List[str]] = None,
        start_x: Optional[float] = None,
        start_y: Optional[float] = None,
        max_nodes: int = 30,
        character_level: int = 60,
        profession_skills: Dict[str, int] = None
    ) -> Optional[FarmingRoute]:
        """
        Calculate optimal farming route for a zone.
        
        Args:
            zone: Zone to create route for
            resource_types: Optional list of resource types to include
            start_x: Starting X coordinate (optional)
            start_y: Starting Y coordinate (optional)
            max_nodes: Maximum nodes to include in route
            character_level: Character level for filtering nodes
            profession_skills: Dict mapping profession to skill level
            
        Returns:
            Optimized farming route or None if no valid route
        """
        if zone not in self.nodes_by_zone or not self.nodes_by_zone[zone]:
            logger.warning(f"No nodes found in zone {zone}")
            return None
            
        # Get all applicable nodes in the zone
        node_ids = self.nodes_by_zone[zone]
        available_nodes = []
        
        for node_id in node_ids:
            node = self.nodes[node_id]
            
            # Filter by resource type if specified
            if resource_types and node.resource_type not in resource_types:
                continue
                
            # Filter by character level
            if node.level_req > character_level:
                continue
                
            # Filter by profession skill if applicable
            if profession_skills and node.resource_type in profession_skills:
                if profession_skills[node.resource_type] < node.skill_req:
                    continue
                    
            # Update node value to ensure it's current
            self._update_node_value(node)
            
            available_nodes.append(node)
            
        if not available_nodes:
            logger.warning(f"No suitable nodes found in zone {zone} for the given criteria")
            return None
            
        # Calculate optimal route using nearest neighbor algorithm with value weighting
        route = FarmingRoute()
        
        # Find starting node - either specified coordinates or highest value node
        current_node = None
        if start_x is not None and start_y is not None:
            # Find closest node to starting coordinates
            closest_dist = float('inf')
            for node in available_nodes:
                dist = math.sqrt((node.x - start_x) ** 2 + (node.y - start_y) ** 2)
                if dist < closest_dist:
                    closest_dist = dist
                    current_node = node
                    
            if current_node:
                route.add_node(current_node, closest_dist)
                available_nodes.remove(current_node)
        else:
            # Start with highest value node
            available_nodes.sort(key=lambda n: n.expected_value, reverse=True)
            current_node = available_nodes[0]
            route.add_node(current_node, 0)
            available_nodes.remove(current_node)
        
        # Build route using nearest neighbor with value weighting
        while available_nodes and len(route.nodes) < max_nodes:
            best_score = float('-inf')
            best_node = None
            best_distance = 0
            
            for node in available_nodes:
                distance = current_node.distance_to(node)
                if distance == float('inf'):
                    continue
                    
                # Score based on value and distance (higher is better)
                # We want high value and low distance
                value_weight = 0.7  # Weight for value component
                distance_weight = 0.3  # Weight for distance component
                
                # Normalize distance to 0-1 scale (closer to 0 is better)
                max_distance = 500  # Maximum reasonable distance in a zone
                normalized_distance = min(distance / max_distance, 1.0)
                
                # Normalize value
                max_value = 500  # Reasonable maximum node value
                normalized_value = min(node.expected_value / max_value, 1.0)
                
                # Calculate score - higher is better
                score = (value_weight * normalized_value) - (distance_weight * normalized_distance)
                
                if score > best_score:
                    best_score = score
                    best_node = node
                    best_distance = distance
            
            if best_node:
                route.add_node(best_node, best_distance)
                current_node = best_node
                available_nodes.remove(best_node)
            else:
                break
        
        return route
        
    def get_best_farming_routes(
        self, 
        character_level: int = 60,
        profession_skills: Optional[Dict[str, int]] = None,
        top_n: int = 3
    ) -> List[FarmingRoute]:
        """
        Get the best farming routes based on gold per hour.
        
        Args:
            character_level: Character level for filtering nodes
            profession_skills: Dict mapping profession to skill level
            top_n: Number of routes to return
            
        Returns:
            List of best farming routes
        """
        if profession_skills is None:
            profession_skills = {}
            
        all_routes = []
        
        # Calculate routes for each zone
        for zone in self.nodes_by_zone.keys():
            # Try different resource type combinations based on professions
            profession_routes = []
            
            # 1. All available nodes
            route = self.calculate_optimal_route(
                zone=zone,
                character_level=character_level,
                profession_skills=profession_skills
            )
            if route:
                profession_routes.append(route)
                
            # 2. Specific resource types if skills are available
            for profession, skill in profession_skills.items():
                if skill > 0:
                    route = self.calculate_optimal_route(
                        zone=zone,
                        resource_types=[profession],
                        character_level=character_level,
                        profession_skills=profession_skills
                    )
                    if route:
                        profession_routes.append(route)
            
            # Add best route for this zone
            if profession_routes:
                best_zone_route = max(profession_routes, key=lambda r: r.gold_per_hour)
                all_routes.append(best_zone_route)
        
        # Sort by gold per hour and return top N
        all_routes.sort(key=lambda r: r.gold_per_hour, reverse=True)
        return all_routes[:top_n]
        
    def adjust_routes_for_competition(
        self,
        routes: List[FarmingRoute],
        competition_data: Dict[str, Dict[int, float]]
    ) -> List[FarmingRoute]:
        """
        Adjust routes based on current competition levels.
        
        Args:
            routes: List of routes to adjust
            competition_data: Nested dict mapping zones to node IDs to competition levels
            
        Returns:
            Adjusted routes
        """
        # Update competition levels in our node database
        for zone, comp_data in competition_data.items():
            self.update_competition_levels(zone, comp_data)
            
        # Recalculate the routes with updated competition data
        adjusted_routes = []
        for route in routes:
            # Get starting point from first node in route
            if route.nodes:
                start_node = route.nodes[0]
                new_route = self.calculate_optimal_route(
                    zone=route.zone,
                    resource_types=list(route.resource_types),
                    start_x=start_node.x,
                    start_y=start_node.y,
                    max_nodes=len(route.nodes)
                )
                if new_route:
                    adjusted_routes.append(new_route)
            
        # Sort by gold per hour
        adjusted_routes.sort(key=lambda r: r.gold_per_hour, reverse=True)
        return adjusted_routes