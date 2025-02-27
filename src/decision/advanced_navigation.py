"""
Advanced Navigation Module

This module provides enhanced navigation capabilities including:
- 3D pathfinding with elevation handling
- Terrain analysis and obstacle avoidance
- Multi-zone routing and dungeon navigation
- Flight path integration
"""

import logging
import random
import math
import numpy as np
import heapq
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.navigation_manager import NavigationManager

class TerrainType:
    """Enum-like class to define terrain types and their movement costs"""
    NORMAL = 1.0       # Default terrain (roads, flat ground)
    GRASS = 1.2        # Slightly slower
    WATER_SHALLOW = 2.0  # Shallow water (wadeable)
    WATER_DEEP = 5.0   # Deep water (swimming)
    MOUNTAIN = 3.0     # Mountain terrain (slow)
    CLIFF = 10.0       # Cliff (very difficult but possible)
    UNWALKABLE = float('inf')  # Completely blocked terrain
    
    # Special terrain types
    FLIGHT_NODE = 0.1  # Flight path node (very fast to traverse via flying)
    INSTANCE_PORTAL = 0.5  # Instance portal (dungeon/raid entrance)

class Position3D:
    """3D position with x, y, z coordinates"""
    
    def __init__(self, x: float, y: float, z: float = 0.0):
        """
        Initialize a 3D position
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate (height/elevation)
        """
        self.x = x
        self.y = y
        self.z = z
    
    def distance_to(self, other: 'Position3D') -> float:
        """
        Calculate distance to another position in 3D space
        
        Args:
            other: Another Position3D object
            
        Returns:
            float: 3D Euclidean distance
        """
        return math.sqrt((self.x - other.x) ** 2 + 
                         (self.y - other.y) ** 2 + 
                         (self.z - other.z) ** 2)
    
    def distance_2d(self, other: 'Position3D') -> float:
        """
        Calculate 2D distance (ignoring elevation)
        
        Args:
            other: Another Position3D object
            
        Returns:
            float: 2D Euclidean distance
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def elevation_difference(self, other: 'Position3D') -> float:
        """
        Calculate elevation difference
        
        Args:
            other: Another Position3D object
            
        Returns:
            float: Absolute elevation difference
        """
        return abs(self.z - other.z)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple representation"""
        return (self.x, self.y, self.z)
    
    def to_tuple_2d(self) -> Tuple[float, float]:
        """Convert to 2D tuple representation (x,y only)"""
        return (self.x, self.y)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float]) -> 'Position3D':
        """Create Position3D from tuple"""
        return cls(coords[0], coords[1], coords[2])
    
    @classmethod
    def from_tuple_2d(cls, coords: Tuple[float, float], z: float = 0.0) -> 'Position3D':
        """Create Position3D from 2D tuple and z coordinate"""
        return cls(coords[0], coords[1], z)
    
    def __eq__(self, other):
        if not isinstance(other, Position3D):
            return False
        return (self.x == other.x and 
                self.y == other.y and 
                self.z == other.z)
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __str__(self):
        return f"Position3D({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"
    
    def __repr__(self):
        return self.__str__()

class TerrainNode:
    """Represents a node in the terrain grid for pathfinding"""
    
    def __init__(self, position: Position3D, terrain_type: float = TerrainType.NORMAL):
        """
        Initialize a terrain node
        
        Args:
            position: 3D position of the node
            terrain_type: Type of terrain (affects movement cost)
        """
        self.position = position
        self.terrain_type = terrain_type
        self.walkable = terrain_type != TerrainType.UNWALKABLE
        self.node_type = "terrain"  # Default node type
        self.metadata = {}  # Additional data (e.g., zone name, instance info)
    
    def get_movement_cost(self) -> float:
        """Get the cost of moving through this terrain"""
        return self.terrain_type
    
    def is_jumpable_from(self, from_node: 'TerrainNode') -> bool:
        """
        Check if this node can be reached by jumping from another node
        
        Args:
            from_node: Origin node for the jump
            
        Returns:
            bool: True if jumpable, False otherwise
        """
        # Basic jump physics check
        # Max horizontal distance for a jump is ~8 units
        # Max height difference for a jump is ~2 units
        
        horizontal_dist = self.position.distance_2d(from_node.position)
        elev_diff = self.position.z - from_node.position.z
        
        # Can jump to higher ground if not too high
        if elev_diff > 0:
            return horizontal_dist < 8 and elev_diff <= 2
        # Can jump/fall to lower ground if not too far horizontally
        else:
            return horizontal_dist < 8
    
    def __str__(self):
        return f"TerrainNode({self.position}, {self.terrain_type:.1f})"
    
    def __repr__(self):
        return self.__str__()

class SpecialNode(TerrainNode):
    """Represents a special navigation node like flight path, portal, etc."""
    
    def __init__(self, position: Position3D, node_type: str, destination=None, name: str = ""):
        """
        Initialize a special node
        
        Args:
            position: 3D position of the node
            node_type: Type of special node (flight_path, portal, etc.)
            destination: Destination of this node (if applicable)
            name: Human-readable name of this node
        """
        super().__init__(position, TerrainType.NORMAL)
        self.node_type = node_type
        self.destination = destination
        self.name = name
        
        # Set appropriate terrain type based on node type
        if node_type == "flight_path":
            self.terrain_type = TerrainType.FLIGHT_NODE
        elif node_type == "portal" or node_type == "instance_portal":
            self.terrain_type = TerrainType.INSTANCE_PORTAL

class ZoneConnection:
    """Represents a connection between two zones"""
    
    def __init__(self, source_zone: str, target_zone: str, 
                 source_pos: Position3D, target_pos: Position3D,
                 connection_type: str = "path"):
        """
        Initialize a zone connection
        
        Args:
            source_zone: Name of the source zone
            target_zone: Name of the target zone
            source_pos: Position in the source zone
            target_pos: Position in the target zone
            connection_type: Type of connection (path, portal, flight)
        """
        self.source_zone = source_zone
        self.target_zone = target_zone
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.connection_type = connection_type
        
        # Movement cost based on connection type
        if connection_type == "flight":
            self.cost = 5.0  # Flight paths take time but are convenient
        elif connection_type == "portal":
            self.cost = 1.0  # Portals are fast
        else:
            # For normal paths, use the distance
            self.cost = source_pos.distance_to(target_pos)

class TerrainMap:
    """Represents a 3D terrain map for a zone with pathfinding capabilities"""
    
    def __init__(self, zone_name: str, resolution: float = 5.0):
        """
        Initialize a terrain map
        
        Args:
            zone_name: Name of the zone this map represents
            resolution: Size of each grid cell in game units
        """
        self.zone_name = zone_name
        self.resolution = resolution
        self.nodes = {}  # Dictionary mapping (x,y,z) -> TerrainNode
        self.special_nodes = []  # List of special nodes (flight paths, etc.)
        self.min_bounds = Position3D(float('inf'), float('inf'), float('inf'))
        self.max_bounds = Position3D(float('-inf'), float('-inf'), float('-inf'))
    
    def add_node(self, node: TerrainNode) -> None:
        """
        Add a node to the terrain map
        
        Args:
            node: TerrainNode to add
        """
        # Update bounds
        self.min_bounds.x = min(self.min_bounds.x, node.position.x)
        self.min_bounds.y = min(self.min_bounds.y, node.position.y)
        self.min_bounds.z = min(self.min_bounds.z, node.position.z)
        
        self.max_bounds.x = max(self.max_bounds.x, node.position.x)
        self.max_bounds.y = max(self.max_bounds.y, node.position.y)
        self.max_bounds.z = max(self.max_bounds.z, node.position.z)
        
        # Add to node dictionary
        key = self._position_to_key(node.position)
        self.nodes[key] = node
        
        # If special node, add to special nodes list
        if node.node_type != "terrain":
            self.special_nodes.append(node)
    
    def get_node_at(self, position: Position3D) -> Optional[TerrainNode]:
        """
        Get the node at a specific position
        
        Args:
            position: Position to query
            
        Returns:
            Optional[TerrainNode]: Node at that position or None
        """
        key = self._position_to_key(position)
        return self.nodes.get(key)
    
    def get_nearest_node(self, position: Position3D) -> TerrainNode:
        """
        Get the nearest node to a position
        
        Args:
            position: Query position
            
        Returns:
            TerrainNode: Nearest node
        """
        # Directly get the node if it exists
        node = self.get_node_at(position)
        if node:
            return node
        
        # Otherwise, find the nearest node
        nearest_node = None
        min_distance = float('inf')
        
        for node in self.nodes.values():
            dist = position.distance_to(node.position)
            if dist < min_distance:
                min_distance = dist
                nearest_node = node
        
        return nearest_node
    
    def get_neighbors(self, node: TerrainNode) -> List[Tuple[TerrainNode, float]]:
        """
        Get neighbor nodes with their costs
        
        Args:
            node: Source node
            
        Returns:
            List[Tuple[TerrainNode, float]]: List of (neighbor, cost) tuples
        """
        neighbors = []
        
        # Check 26 neighbors in 3D grid (8 in 2D plane, and those 8 at higher and lower z)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip the node itself
                    
                    # Calculate neighbor position
                    neighbor_pos = Position3D(
                        node.position.x + dx * self.resolution,
                        node.position.y + dy * self.resolution,
                        node.position.z + dz * self.resolution
                    )
                    
                    # Get the node at that position
                    neighbor = self.get_node_at(neighbor_pos)
                    
                    if neighbor and neighbor.walkable:
                        # Calculate movement cost
                        base_cost = node.position.distance_to(neighbor_pos)
                        terrain_cost = neighbor.get_movement_cost()
                        
                        # Add additional cost for elevation changes
                        elev_diff = abs(neighbor_pos.z - node.position.z)
                        elevation_cost = elev_diff * 1.5  # Climbing/descending is more costly
                        
                        total_cost = base_cost * terrain_cost + elevation_cost
                        neighbors.append((neighbor, total_cost))
        
        # Check for jumpable nodes (further away but reachable by jumping)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                # Skip nearby nodes already checked
                if abs(dx) <= 1 and abs(dy) <= 1:
                    continue
                
                for dz in [-2, -1, 0, 1, 2]:  # Check vertical range for jumps
                    # Skip if too far for a jump
                    if (dx**2 + dy**2) > 9:  # Max jump distance squared
                        continue
                    
                    # Calculate potential jump target position
                    jump_pos = Position3D(
                        node.position.x + dx * self.resolution,
                        node.position.y + dy * self.resolution,
                        node.position.z + dz * self.resolution
                    )
                    
                    jump_node = self.get_node_at(jump_pos)
                    
                    if jump_node and jump_node.walkable and jump_node.is_jumpable_from(node):
                        # Jumping has a fixed cost plus distance
                        jump_cost = 5.0 + node.position.distance_to(jump_pos)
                        neighbors.append((jump_node, jump_cost))
        
        return neighbors
    
    def _position_to_key(self, position: Position3D) -> Tuple[int, int, int]:
        """
        Convert a position to a grid key
        
        Args:
            position: Position to convert
            
        Returns:
            Tuple[int, int, int]: Grid key
        """
        # Convert to grid coordinates
        grid_x = int(position.x // self.resolution)
        grid_y = int(position.y // self.resolution)
        grid_z = int(position.z // self.resolution)
        
        return (grid_x, grid_y, grid_z)
    
    def get_nearest_special_node(self, position: Position3D, node_type: str = None) -> Optional[SpecialNode]:
        """
        Get the nearest special node of a specific type
        
        Args:
            position: Query position
            node_type: Type of special node to find (None for any)
            
        Returns:
            Optional[SpecialNode]: Nearest special node or None
        """
        nearest_node = None
        min_distance = float('inf')
        
        for node in self.special_nodes:
            if node_type and node.node_type != node_type:
                continue
                
            dist = position.distance_to(node.position)
            if dist < min_distance:
                min_distance = dist
                nearest_node = node
        
        return nearest_node
    
    def add_water_body(self, center: Position3D, radius: float, depth: float) -> None:
        """
        Add a circular water body to the terrain
        
        Args:
            center: Center position of the water body
            radius: Radius of the water body
            depth: Depth of the water (affects terrain type)
        """
        # Determine terrain type based on depth
        terrain_type = TerrainType.WATER_SHALLOW if depth < 2.0 else TerrainType.WATER_DEEP
        
        # Calculate grid bounds
        min_x = int((center.x - radius) // self.resolution)
        max_x = int((center.x + radius) // self.resolution)
        min_y = int((center.y - radius) // self.resolution)
        max_y = int((center.y + radius) // self.resolution)
        
        # Add water nodes
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                pos = Position3D(
                    x * self.resolution,
                    y * self.resolution,
                    center.z - depth  # Water surface is at center.z
                )
                
                # Check if within radius
                if pos.distance_2d(Position3D(center.x, center.y, 0)) <= radius:
                    self.add_node(TerrainNode(pos, terrain_type))

class AdvancedNavigationManager:
    """
    Advanced navigation manager that extends the basic NavigationManager
    with 3D pathfinding, terrain analysis, and multi-zone navigation
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge, basic_nav: NavigationManager):
        """
        Initialize the AdvancedNavigationManager
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
            basic_nav: Basic navigation manager instance
        """
        self.logger = logging.getLogger("wow_ai.decision.advanced_navigation")
        self.config = config
        self.knowledge = knowledge
        self.basic_nav = basic_nav
        
        # Terrain maps for each zone
        self.terrain_maps = {}  # zone_name -> TerrainMap
        
        # Zone connections
        self.zone_connections = []  # List of ZoneConnection objects
        
        # Navigation states
        self.current_zone = ""
        self.current_position = None
        self.destination_zone = ""
        self.destination_position = None
        
        # Path cache
        self.path_cache = {}  # (start_pos, end_pos, zone) -> path
        self.path_cache_ttl = 300  # Time-to-live in seconds
        self.path_cache_timestamps = {}  # (start_pos, end_pos, zone) -> timestamp
        
        # Flight path network
        self.flight_paths = {}  # zone -> list of flight points
        self.flight_connections = {}  # (source, target) -> is_connected
        
        # Dungeon entrances
        self.dungeon_entrances = {}  # dungeon_name -> (zone, position)
        
        self._load_terrain_data()
        self._load_flight_paths()
        self._load_dungeon_data()
        
        self.logger.info("AdvancedNavigationManager initialized")
    
    def _load_terrain_data(self) -> None:
        """
        Load terrain data for all zones
        """
        # In a real implementation, this would load actual terrain data
        # Here we'll create a simple flat terrain for demonstration
        
        # Check if we already have terrain maps in knowledge
        known_zones = self.knowledge.zones.keys()
        
        for zone_name in known_zones:
            self.logger.info(f"Initializing terrain map for {zone_name}")
            terrain_map = TerrainMap(zone_name)
            
            # Get zone bounds from knowledge if available
            zone_data = self.knowledge.zones.get(zone_name, {})
            zone_bounds = zone_data.get("bounds", {})
            
            min_x = zone_bounds.get("min_x", 0)
            max_x = zone_bounds.get("max_x", 1000)
            min_y = zone_bounds.get("min_y", 0)
            max_y = zone_bounds.get("max_y", 1000)
            
            # Add basic terrain (just a flat grid for now)
            # In a real implementation, this would include actual elevation data
            for x in range(min_x, max_x, 50):  # Use larger steps for efficiency
                for y in range(min_y, max_y, 50):
                    pos = Position3D(x, y, 0)
                    terrain_map.add_node(TerrainNode(pos, TerrainType.NORMAL))
            
            # Add POIs as nodes
            for poi in zone_data.get("points_of_interest", []):
                poi_name = poi.get("name", "")
                poi_pos = poi.get("position", [0, 0])
                pos = Position3D(poi_pos[0], poi_pos[1], 0)
                
                # Create a normal terrain node for the POI
                node = TerrainNode(pos, TerrainType.NORMAL)
                node.metadata["name"] = poi_name
                node.metadata["type"] = "poi"
                terrain_map.add_node(node)
            
            self.terrain_maps[zone_name] = terrain_map
        
        # Add zone connections
        self._init_zone_connections()
    
    def _init_zone_connections(self) -> None:
        """
        Initialize zone connections based on knowledge data
        """
        # Check known zones for connections
        for zone_name, zone_data in self.knowledge.zones.items():
            neighbors = zone_data.get("neighbors", [])
            
            for neighbor in neighbors:
                # Skip if we don't have data for the neighbor zone
                if neighbor not in self.knowledge.zones:
                    continue
                
                # Find connection points between zones
                # In a real implementation, these would be accurate boundary points
                # For simplicity, we'll use arbitrary positions
                
                # Get the nearest POIs to the border
                source_pois = zone_data.get("points_of_interest", [])
                target_pois = self.knowledge.zones[neighbor].get("points_of_interest", [])
                
                if not source_pois or not target_pois:
                    continue
                
                # Find closest POIs between zones (simplified approximation)
                min_distance = float('inf')
                closest_source = None
                closest_target = None
                
                for source_poi in source_pois:
                    source_pos = source_poi.get("position", [0, 0])
                    
                    for target_poi in target_pois:
                        target_pos = target_poi.get("position", [0, 0])
                        
                        dx = source_pos[0] - target_pos[0]
                        dy = source_pos[1] - target_pos[1]
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_source = source_pos
                            closest_target = target_pos
                
                if closest_source and closest_target:
                    # Create a zone connection
                    source_position = Position3D(closest_source[0], closest_source[1], 0)
                    target_position = Position3D(closest_target[0], closest_target[1], 0)
                    
                    connection = ZoneConnection(
                        source_zone=zone_name,
                        target_zone=neighbor,
                        source_pos=source_position,
                        target_pos=target_position,
                        connection_type="path"
                    )
                    
                    self.zone_connections.append(connection)
                    self.logger.info(f"Added zone connection: {zone_name} <-> {neighbor}")
    
    def _load_flight_paths(self) -> None:
        """
        Load flight path data
        """
        # Check known zones for flight paths
        for zone_name, zone_data in self.knowledge.zones.items():
            flight_paths = zone_data.get("flight_paths", [])
            
            if not flight_paths:
                continue
            
            self.flight_paths[zone_name] = []
            
            for path in flight_paths:
                # In a real implementation, this would include actual flight path data
                # For now, we'll just create a dummy flight path node
                
                # Get POI for this flight path if available
                flight_point_pos = [0, 0]
                flight_point_name = path
                
                for poi in zone_data.get("points_of_interest", []):
                    if poi.get("name", "") == path:
                        flight_point_pos = poi.get("position", [0, 0])
                        break
                
                position = Position3D(flight_point_pos[0], flight_point_pos[1], 0)
                
                # Create flight path node
                flight_node = SpecialNode(
                    position=position,
                    node_type="flight_path",
                    name=flight_point_name
                )
                
                # Add to terrain map
                if zone_name in self.terrain_maps:
                    self.terrain_maps[zone_name].add_node(flight_node)
                
                # Add to flight paths
                self.flight_paths[zone_name].append(flight_node)
        
        # Set up flight connections (which flight paths are connected)
        # In a real implementation, this would use actual flight path data
        # For simplicity, we'll assume all flight paths in neighboring zones are connected
        
        for zone_name in self.flight_paths:
            for neighbor in self.knowledge.zones.get(zone_name, {}).get("neighbors", []):
                if neighbor in self.flight_paths:
                    for source_node in self.flight_paths[zone_name]:
                        for target_node in self.flight_paths[neighbor]:
                            key = (source_node.name, target_node.name)
                            self.flight_connections[key] = True
    
    def _load_dungeon_data(self) -> None:
        """
        Load dungeon entrance data
        """
        # Check if we have dungeon data
        if not hasattr(self.knowledge, 'instances'):
            return
        
        # Process each dungeon/raid instance
        for instance_id, instance_data in self.knowledge.instances.items():
            instance_name = instance_data.get("name", instance_id)
            
            # Look for entrance location
            entrance = instance_data.get("entrance", {})
            entrance_zone = entrance.get("zone", "")
            entrance_pos = entrance.get("position", [0, 0, 0])
            
            if not entrance_zone or not entrance_pos:
                continue
            
            # Create 3D position
            if len(entrance_pos) >= 3:
                position = Position3D(entrance_pos[0], entrance_pos[1], entrance_pos[2])
            else:
                position = Position3D(entrance_pos[0], entrance_pos[1], 0)
            
            # Add to dungeon entrances
            self.dungeon_entrances[instance_name.lower()] = (entrance_zone, position)
            
            # Add special node to terrain map
            if entrance_zone in self.terrain_maps:
                entrance_node = SpecialNode(
                    position=position,
                    node_type="instance_portal",
                    name=instance_name,
                    destination=instance_id
                )
                self.terrain_maps[entrance_zone].add_node(entrance_node)
                self.logger.info(f"Added dungeon entrance for {instance_name} in {entrance_zone}")
    
    def navigate_to(self, state: GameState, target_position: Tuple[float, float, float], 
                   target_zone: str = None) -> List[Dict]:
        """
        Navigate to a target position, possibly in another zone
        
        Args:
            state: Current game state
            target_position: Target position (x, y, z)
            target_zone: Target zone name (None if same as current)
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Get current zone and position
        current_zone = state.current_zone if hasattr(state, "current_zone") else ""
        
        if not current_zone:
            self.logger.warning("Current zone unknown, falling back to basic navigation")
            target_position_2d = (target_position[0], target_position[1])
            return self.basic_nav.generate_navigation_plan(state, target_position_2d)
        
        # Get current position
        current_position = None
        if hasattr(state, "player_position"):
            if len(state.player_position) >= 3:
                current_position = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    state.player_position[2]
                )
            else:
                current_position = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    0
                )
        
        if not current_position:
            self.logger.warning("Current position unknown, falling back to basic navigation")
            target_position_2d = (target_position[0], target_position[1])
            return self.basic_nav.generate_navigation_plan(state, target_position_2d)
        
        # Convert target position to Position3D
        if len(target_position) >= 3:
            target_pos = Position3D(
                target_position[0],
                target_position[1],
                target_position[2]
            )
        else:
            target_pos = Position3D(
                target_position[0],
                target_position[1],
                0
            )
        
        # Set target zone (default to current if not specified)
        target_zone = target_zone or current_zone
        
        # Update current navigation state
        self.current_zone = current_zone
        self.current_position = current_position
        self.destination_zone = target_zone
        self.destination_position = target_pos
        
        # Check if same zone navigation or cross-zone
        if current_zone == target_zone:
            # Same zone navigation
            return self._navigate_within_zone(current_position, target_pos, current_zone)
        else:
            # Cross-zone navigation
            return self._navigate_cross_zone(current_position, target_pos, current_zone, target_zone)
    
    def _navigate_within_zone(self, start_pos: Position3D, end_pos: Position3D, 
                            zone: str) -> List[Dict]:
        """
        Navigate within a single zone
        
        Args:
            start_pos: Starting position
            end_pos: Target position
            zone: Zone name
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Check path cache
        cache_key = (start_pos.to_tuple(), end_pos.to_tuple(), zone)
        if cache_key in self.path_cache:
            # Check if cache entry is still valid
            timestamp = self.path_cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.path_cache_ttl:
                # Use cached path
                self.logger.info(f"Using cached path for {zone}: {start_pos} -> {end_pos}")
                return self._path_to_actions(self.path_cache[cache_key], zone)
        
        # Get terrain map for zone
        terrain_map = self.terrain_maps.get(zone)
        if not terrain_map:
            self.logger.warning(f"No terrain map for zone {zone}, falling back to basic navigation")
            return self.basic_nav.generate_navigation_plan(None, end_pos.to_tuple_2d())
        
        # Get terrain nodes for start and end positions
        start_node = terrain_map.get_nearest_node(start_pos)
        end_node = terrain_map.get_nearest_node(end_pos)
        
        if not start_node or not end_node:
            self.logger.warning(f"Could not find start or end node in zone {zone}")
            return self.basic_nav.generate_navigation_plan(None, end_pos.to_tuple_2d())
        
        # Use A* pathfinding to find path
        path = self._find_path_astar(start_node, end_node, terrain_map)
        
        if not path:
            self.logger.warning(f"No path found in zone {zone}, falling back to basic navigation")
            return self.basic_nav.generate_navigation_plan(None, end_pos.to_tuple_2d())
        
        # Cache the path
        self.path_cache[cache_key] = [node.position for node in path]
        self.path_cache_timestamps[cache_key] = time.time()
        
        # Convert path to actions
        return self._path_to_actions(path, zone)
    
    def _navigate_cross_zone(self, start_pos: Position3D, end_pos: Position3D,
                           start_zone: str, end_zone: str) -> List[Dict]:
        """
        Navigate across multiple zones
        
        Args:
            start_pos: Starting position
            end_pos: Target position
            start_zone: Starting zone
            end_zone: Target zone
            
        Returns:
            List[Dict]: Navigation actions
        """
        # First, check if there's a direct flight path
        flight_path = self._get_flight_path(start_pos, end_pos, start_zone, end_zone)
        if flight_path:
            return flight_path
        
        # Second, find a multi-zone path
        zone_path = self._find_zone_path(start_zone, end_zone)
        
        if not zone_path:
            self.logger.warning(f"No path found between zones {start_zone} and {end_zone}")
            return []
        
        # Create a path through each zone
        full_path = []
        current_pos = start_pos
        
        for i in range(len(zone_path) - 1):
            source_zone = zone_path[i]
            target_zone = zone_path[i + 1]
            
            # Find connection between zones
            connection = self._get_zone_connection(source_zone, target_zone)
            
            if not connection:
                self.logger.warning(f"No connection found between {source_zone} and {target_zone}")
                continue
            
            # Navigate to connection point in source zone
            if i == 0:  # First zone (starting zone)
                zone_actions = self._navigate_within_zone(current_pos, connection.source_pos, source_zone)
            else:
                # We're already at the target position of the previous connection
                zone_actions = []
            
            full_path.extend(zone_actions)
            
            # Add zone transition action
            full_path.append({
                "type": "zone_transition",
                "source_zone": source_zone,
                "target_zone": target_zone,
                "description": f"Cross from {source_zone} to {target_zone}"
            })
            
            current_pos = connection.target_pos
        
        # Finally, navigate to destination in final zone
        final_actions = self._navigate_within_zone(current_pos, end_pos, end_zone)
        full_path.extend(final_actions)
        
        return full_path
    
    def _find_path_astar(self, start_node: TerrainNode, goal_node: TerrainNode, 
                       terrain_map: TerrainMap) -> List[TerrainNode]:
        """
        Find a path using A* algorithm
        
        Args:
            start_node: Starting node
            goal_node: Target node
            terrain_map: Terrain map to use
            
        Returns:
            List[TerrainNode]: Path as a list of nodes
        """
        # Initialize data structures
        open_set = []  # Priority queue
        closed_set = set()
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start_node], id(start_node), start_node))
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current == goal_node:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                path.reverse()
                return path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Check neighbors
            for neighbor, cost in terrain_map.get_neighbors(current):
                # Skip if neighbor in closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g score
                tentative_g = g_score[current] + cost
                
                # Check if this path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal_node)
                    
                    # Add to open set if not already there
                    if neighbor not in [n for _, _, n in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
        
        # If we get here, no path was found
        self.logger.warning("A* pathfinding failed to find a path")
        return []
    
    def _heuristic(self, a: TerrainNode, b: TerrainNode) -> float:
        """
        Heuristic function for A* pathfinding (3D Euclidean distance)
        
        Args:
            a: First node
            b: Second node
            
        Returns:
            float: Estimated distance between nodes
        """
        return a.position.distance_to(b.position)
    
    def _path_to_actions(self, path: List[Union[TerrainNode, Position3D]], zone: str) -> List[Dict]:
        """
        Convert a path to navigation actions
        
        Args:
            path: Path as a list of nodes or positions
            zone: Zone name
            
        Returns:
            List[Dict]: Navigation actions
        """
        actions = []
        
        # Check if the path is empty
        if not path:
            return actions
        
        # Check if the first element is a TerrainNode or Position3D
        if isinstance(path[0], TerrainNode):
            positions = [node.position for node in path]
        else:
            positions = path
        
        # Optimize the path by reducing unnecessary waypoints
        optimized = self._optimize_path(positions)
        
        # Convert to actions
        for i, pos in enumerate(optimized):
            # Check if it's a special location
            description = f"Move to waypoint {i+1}/{len(optimized)}"
            action_type = "move"
            metadata = {}
            
            # Check terrain maps for special nodes
            if zone in self.terrain_maps:
                node = self.terrain_maps[zone].get_node_at(pos)
                
                if node and node.node_type != "terrain":
                    # Special node actions
                    if node.node_type == "flight_path":
                        action_type = "flight_path"
                        description = f"Use flight path at {node.name}"
                        metadata["flight_point"] = node.name
                    elif node.node_type == "instance_portal":
                        action_type = "instance_portal"
                        description = f"Enter {node.name} instance"
                        metadata["instance"] = node.name
                    elif node.node_type == "portal":
                        action_type = "portal"
                        description = f"Use portal to {node.destination}"
                        metadata["destination"] = node.destination
            
            # Create the action
            action = {
                "type": action_type,
                "position": pos.to_tuple(),
                "description": description
            }
            
            # Add metadata if present
            if metadata:
                action["metadata"] = metadata
            
            actions.append(action)
        
        return actions
    
    def _optimize_path(self, path: List[Position3D]) -> List[Position3D]:
        """
        Optimize a path by removing unnecessary waypoints
        
        Args:
            path: Original path as list of positions
            
        Returns:
            List[Position3D]: Optimized path
        """
        if len(path) <= 2:
            return path
        
        optimized = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = optimized[-1]
            current = path[i]
            next_pos = path[i + 1]
            
            # Calculate vectors
            v1 = Position3D(current.x - prev.x, current.y - prev.y, current.z - prev.z)
            v2 = Position3D(next_pos.x - current.x, next_pos.y - current.y, next_pos.z - current.z)
            
            # Normalize vectors (in 3D)
            len_v1 = math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2)
            len_v2 = math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2)
            
            if len_v1 > 0 and len_v2 > 0:
                v1_norm = Position3D(v1.x / len_v1, v1.y / len_v1, v1.z / len_v1)
                v2_norm = Position3D(v2.x / len_v2, v2.y / len_v2, v2.z / len_v2)
                
                # Calculate dot product to get cosine of angle
                dot_product = (v1_norm.x * v2_norm.x + 
                               v1_norm.y * v2_norm.y + 
                               v1_norm.z * v2_norm.z)
                
                # If directions change significantly or elevation changes, keep the waypoint
                if dot_product < 0.9 or abs(current.z - prev.z) > 0.5:
                    optimized.append(current)
            else:
                optimized.append(current)
        
        optimized.append(path[-1])
        return optimized
    
    def _get_zone_connection(self, source_zone: str, target_zone: str) -> Optional[ZoneConnection]:
        """
        Get a connection between two zones
        
        Args:
            source_zone: Source zone name
            target_zone: Target zone name
            
        Returns:
            Optional[ZoneConnection]: Zone connection or None
        """
        for connection in self.zone_connections:
            if (connection.source_zone == source_zone and connection.target_zone == target_zone):
                return connection
            # Check reverse direction as well
            elif (connection.source_zone == target_zone and connection.target_zone == source_zone):
                # Create a reversed connection
                return ZoneConnection(
                    source_zone=target_zone,
                    target_zone=source_zone,
                    source_pos=connection.target_pos,
                    target_pos=connection.source_pos,
                    connection_type=connection.connection_type
                )
        
        return None
    
    def _find_zone_path(self, start_zone: str, end_zone: str) -> List[str]:
        """
        Find a path through zones
        
        Args:
            start_zone: Starting zone
            end_zone: Target zone
            
        Returns:
            List[str]: Path as a list of zone names
        """
        # Simple BFS to find a path through zones
        visited = set()
        queue = [[start_zone]]
        
        while queue:
            path = queue.pop(0)
            zone = path[-1]
            
            if zone == end_zone:
                return path
            
            if zone not in visited:
                visited.add(zone)
                
                # Get neighboring zones
                neighbors = set()
                
                # Add directly connected neighbors from knowledge
                if zone in self.knowledge.zones:
                    neighbors.update(self.knowledge.zones[zone].get("neighbors", []))
                
                # Add zones connected by flight paths
                if zone in self.flight_paths:
                    for flight_node in self.flight_paths[zone]:
                        for other_zone, flight_nodes in self.flight_paths.items():
                            if other_zone == zone:
                                continue
                                
                            for other_node in flight_nodes:
                                if (flight_node.name, other_node.name) in self.flight_connections:
                                    neighbors.add(other_zone)
                
                # Add unvisited neighbors to queue
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(path + [neighbor])
        
        return []
    
    def _get_flight_path(self, start_pos: Position3D, end_pos: Position3D,
                      start_zone: str, end_zone: str) -> Optional[List[Dict]]:
        """
        Get a flight path route between zones
        
        Args:
            start_pos: Starting position
            end_pos: Target position
            start_zone: Starting zone
            end_zone: Target zone
            
        Returns:
            Optional[List[Dict]]: Flight path actions or None
        """
        # Check if both zones have flight paths
        if start_zone not in self.flight_paths or end_zone not in self.flight_paths:
            return None
        
        # Find nearest flight points in each zone
        if start_zone in self.terrain_maps:
            start_flight = self.terrain_maps[start_zone].get_nearest_special_node(
                start_pos, node_type="flight_path")
        else:
            start_flight = None
            
        if end_zone in self.terrain_maps:
            end_flight = self.terrain_maps[end_zone].get_nearest_special_node(
                end_pos, node_type="flight_path")
        else:
            end_flight = None
        
        if not start_flight or not end_flight:
            return None
        
        # Check if these flight points are connected
        if (start_flight.name, end_flight.name) not in self.flight_connections:
            return None
        
        # Create a flight path route
        actions = []
        
        # 1. Navigate to flight point in starting zone
        to_flight_actions = self._navigate_within_zone(start_pos, start_flight.position, start_zone)
        actions.extend(to_flight_actions)
        
        # 2. Take flight path
        actions.append({
            "type": "use_flight_path",
            "source": start_flight.name,
            "destination": end_flight.name,
            "description": f"Take flight from {start_flight.name} to {end_flight.name}"
        })
        
        # 3. Navigate from flight point to destination in end zone
        from_flight_actions = self._navigate_within_zone(end_flight.position, end_pos, end_zone)
        actions.extend(from_flight_actions)
        
        return actions
    
    def navigate_to_instance(self, state: GameState, instance_name: str) -> List[Dict]:
        """
        Navigate to a dungeon or raid instance
        
        Args:
            state: Current game state
            instance_name: Name of the instance
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Normalize instance name
        instance_name = instance_name.lower()
        
        # Find instance entrance
        if instance_name not in self.dungeon_entrances:
            self.logger.warning(f"Unknown instance: {instance_name}")
            return []
        
        # Get entrance data
        entrance_zone, entrance_pos = self.dungeon_entrances[instance_name]
        
        # Navigate to entrance
        return self.navigate_to(state, entrance_pos.to_tuple(), entrance_zone)
    
    def detect_and_avoid_obstacles(self, state: GameState, current_path: List[Position3D]) -> List[Position3D]:
        """
        Detect and avoid dynamic obstacles in the current path
        
        Args:
            state: Current game state
            current_path: Current navigation path
            
        Returns:
            List[Position3D]: Updated path avoiding obstacles
        """
        # Skip if no path
        if not current_path:
            return current_path
        
        # Check if we have obstacle data
        if not hasattr(state, "obstacles") or not state.obstacles:
            return current_path
        
        # Convert obstacles to Position3D
        obstacles = []
        for obs in state.obstacles:
            if len(obs) >= 3:
                pos = Position3D(obs[0], obs[1], obs[2])
            else:
                pos = Position3D(obs[0], obs[1], 0)
            
            # Add radius information if available
            radius = 2.0  # Default radius
            if isinstance(obs, dict) and "radius" in obs:
                radius = obs["radius"]
            
            obstacles.append((pos, radius))
        
        # Check if any obstacles intersect our path
        has_intersection = False
        for i in range(len(current_path) - 1):
            seg_start = current_path[i]
            seg_end = current_path[i+1]
            
            for obs_pos, obs_radius in obstacles:
                # Check distance from obstacle to path segment
                dist = self._point_to_segment_distance(obs_pos, seg_start, seg_end)
                
                if dist < obs_radius:
                    has_intersection = True
                    break
            
            if has_intersection:
                break
        
        # If no intersection, keep current path
        if not has_intersection:
            return current_path
        
        # Otherwise, recalculate path
        # In a real implementation, this would use A* with dynamic obstacle avoidance
        # For simplicity, we'll just add waypoints around obstacles
        
        new_path = [current_path[0]]
        
        for i in range(len(current_path) - 1):
            seg_start = current_path[i]
            seg_end = current_path[i+1]
            
            has_obstacle = False
            for obs_pos, obs_radius in obstacles:
                dist = self._point_to_segment_distance(obs_pos, seg_start, seg_end)
                
                if dist < obs_radius:
                    has_obstacle = True
                    
                    # Generate detour around obstacle
                    detour_points = self._generate_detour(seg_start, seg_end, obs_pos, obs_radius)
                    new_path.extend(detour_points)
                    break
            
            if not has_obstacle:
                new_path.append(seg_end)
        
        return new_path
    
    def _point_to_segment_distance(self, point: Position3D, seg_start: Position3D, 
                                 seg_end: Position3D) -> float:
        """
        Calculate minimum distance from a point to a line segment
        
        Args:
            point: Query point
            seg_start: Segment start point
            seg_end: Segment end point
            
        Returns:
            float: Minimum distance
        """
        # Calculate vectors
        segment = Position3D(seg_end.x - seg_start.x, seg_end.y - seg_start.y, seg_end.z - seg_start.z)
        point_to_start = Position3D(point.x - seg_start.x, point.y - seg_start.y, point.z - seg_start.z)
        
        # Calculate squared length of segment
        seg_length_sq = segment.x**2 + segment.y**2 + segment.z**2
        
        if seg_length_sq == 0:
            # Segment is a point
            return point.distance_to(seg_start)
        
        # Calculate projection of point_to_start onto segment
        t = max(0, min(1, (point_to_start.x * segment.x + 
                           point_to_start.y * segment.y + 
                           point_to_start.z * segment.z) / seg_length_sq))
        
        # Calculate closest point on segment
        closest = Position3D(
            seg_start.x + t * segment.x,
            seg_start.y + t * segment.y,
            seg_start.z + t * segment.z
        )
        
        # Return distance
        return point.distance_to(closest)
    
    def _generate_detour(self, start: Position3D, end: Position3D, 
                       obstacle: Position3D, radius: float) -> List[Position3D]:
        """
        Generate a detour around an obstacle
        
        Args:
            start: Start position
            end: End position
            obstacle: Obstacle position
            radius: Obstacle radius
            
        Returns:
            List[Position3D]: Detour waypoints
        """
        # Calculate vector from start to end
        direction = Position3D(end.x - start.x, end.y - start.y, end.z - start.z)
        length = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        
        if length == 0:
            return [end]
        
        # Normalize direction
        direction = Position3D(direction.x / length, direction.y / length, direction.z / length)
        
        # Calculate perpendicular vector (in 2D, ignore z)
        perp = Position3D(-direction.y, direction.x, 0)
        
        # Calculate amount to offset (radius plus buffer)
        offset_distance = radius * 1.5
        
        # Create detour points
        midpoint = Position3D(
            (start.x + end.x) / 2,
            (start.y + end.y) / 2,
            (start.z + end.z) / 2
        )
        
        # Check which side to go around
        vec_to_obstacle = Position3D(
            obstacle.x - midpoint.x,
            obstacle.y - midpoint.y,
            obstacle.z - midpoint.z
        )
        
        # Dot product to determine side
        side = vec_to_obstacle.x * perp.x + vec_to_obstacle.y * perp.y
        
        # Go around the opposite side of the obstacle
        if side > 0:
            offset_distance = -offset_distance
        
        # Create waypoint with offset
        waypoint = Position3D(
            midpoint.x + perp.x * offset_distance,
            midpoint.y + perp.y * offset_distance,
            midpoint.z  # Keep same elevation
        )
        
        return [waypoint]