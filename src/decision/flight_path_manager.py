"""
Flight Path Manager Module

This module manages in-game flight paths for efficient travel:
- Flight path discovery and tracking
- Optimal flight path selection
- Multi-hop flight routing
- Cost/time estimation
"""

import logging
import math
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.advanced_navigation import Position3D, SpecialNode, ZoneConnection

class FlightNode:
    """Represents a flight path node (flight master location)"""
    
    def __init__(self, name: str, position: Position3D, zone: str):
        """
        Initialize a flight node
        
        Args:
            name: Name of the flight point
            position: Position of the flight master
            zone: Zone name
        """
        self.name = name
        self.position = position
        self.zone = zone
        self.discovered = False
        self.connections = {}  # name -> FlightConnection
    
    def add_connection(self, target: 'FlightNode', cost: float) -> None:
        """
        Add a connection to another flight node
        
        Args:
            target: Target flight node
            cost: Cost (time) of flight
        """
        self.connections[target.name] = FlightConnection(self, target, cost)
    
    def get_connections(self) -> List['FlightConnection']:
        """Get all connections from this node"""
        return list(self.connections.values())
    
    def __str__(self) -> str:
        return f"FlightNode({self.name}, {self.zone})"
    
    def __repr__(self) -> str:
        return self.__str__()

class FlightConnection:
    """Represents a connection between two flight path nodes"""
    
    def __init__(self, source: FlightNode, target: FlightNode, cost: float):
        """
        Initialize a flight connection
        
        Args:
            source: Source flight node
            target: Target flight node
            cost: Cost (time) of flight
        """
        self.source = source
        self.target = target
        self.cost = cost
        self.discovered = False

class FlightPathManager:
    """
    Manages flight paths for efficient zone travel
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the FlightPathManager
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.flight_path_manager")
        self.config = config
        self.knowledge = knowledge
        
        # Flight nodes by name
        self.flight_nodes = {}  # name -> FlightNode
        
        # Flight nodes by zone
        self.zone_flight_nodes = {}  # zone -> [FlightNode]
        
        # Initialize flight network from knowledge
        self._init_flight_network()
        
        self.logger.info("FlightPathManager initialized")
    
    def _init_flight_network(self) -> None:
        """Initialize flight network from knowledge base"""
        # Check known zones for flight paths
        for zone_name, zone_data in self.knowledge.zones.items():
            flight_paths = zone_data.get("flight_paths", [])
            
            if not flight_paths:
                continue
            
            # Create flight nodes for this zone
            self.zone_flight_nodes[zone_name] = []
            
            for path_name in flight_paths:
                # Get position for this flight path if available
                flight_point_pos = [0, 0, 0]
                
                for poi in zone_data.get("points_of_interest", []):
                    if poi.get("name", "") == path_name:
                        pos = poi.get("position", [0, 0])
                        if len(pos) >= 2:
                            flight_point_pos = [pos[0], pos[1], 0]
                        break
                
                # Create flight node
                position = Position3D(
                    flight_point_pos[0],
                    flight_point_pos[1],
                    flight_point_pos[2] if len(flight_point_pos) > 2 else 0
                )
                
                node = FlightNode(path_name, position, zone_name)
                
                # Mark node as discovered if it's a starter zone flight path
                faction = zone_data.get("faction", "neutral")
                starter_zones = {
                    "alliance": ["elwynn_forest", "dun_morogh", "teldrassil"],
                    "horde": ["durotar", "mulgore", "tirisfal_glades"]
                }
                
                if faction in starter_zones and zone_name in starter_zones[faction]:
                    node.discovered = True
                
                # Add to collections
                self.flight_nodes[path_name] = node
                self.zone_flight_nodes[zone_name].append(node)
        
        # Create connections between flight nodes
        self._init_flight_connections()
    
    def _init_flight_connections(self) -> None:
        """Initialize flight connections based on zone adjacency"""
        # In a real implementation, this would use actual flight path data
        # For simplicity, we'll create connections between all flight points in adjacent zones
        
        for zone_name, flight_nodes in self.zone_flight_nodes.items():
            # Get adjacent zones
            adjacent_zones = self.knowledge.zones.get(zone_name, {}).get("neighbors", [])
            
            # Create connections to flight points in adjacent zones
            for node in flight_nodes:
                # Connect to other flight points in same zone
                for other_node in flight_nodes:
                    if node != other_node:
                        # Calculate flight cost (based on distance)
                        distance = node.position.distance_to(other_node.position)
                        cost = max(1.0, distance / 20.0)  # 1 minute minimum, then 1 min per 20 distance units
                        
                        node.add_connection(other_node, cost)
                
                # Connect to flight points in adjacent zones
                for adj_zone in adjacent_zones:
                    if adj_zone in self.zone_flight_nodes:
                        for adj_node in self.zone_flight_nodes[adj_zone]:
                            # Calculate flight cost (based on distance)
                            distance = node.position.distance_to(adj_node.position)
                            cost = max(1.5, distance / 15.0)  # 1.5 minutes minimum, then 1 min per 15 distance units (slower for cross-zone)
                            
                            node.add_connection(adj_node, cost)
        
        self.logger.info(f"Initialized flight network with {len(self.flight_nodes)} nodes")
    
    def update_discovered_flights(self, flight_points: List[str]) -> None:
        """
        Update discovered flight points
        
        Args:
            flight_points: List of discovered flight point names
        """
        for fp_name in flight_points:
            if fp_name in self.flight_nodes:
                self.flight_nodes[fp_name].discovered = True
                self.logger.info(f"Discovered flight point: {fp_name}")
    
    def get_nearest_flight_point(self, position: Position3D, zone: str) -> Optional[FlightNode]:
        """
        Get the nearest flight point in a zone
        
        Args:
            position: Current position
            zone: Current zone
            
        Returns:
            Optional[FlightNode]: Nearest flight point or None
        """
        if zone not in self.zone_flight_nodes:
            return None
        
        # Find the nearest flight point
        nearest_node = None
        min_distance = float('inf')
        
        for node in self.zone_flight_nodes[zone]:
            distance = position.distance_to(node.position)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def find_flight_path(self, start_zone: str, start_pos: Position3D,
                       end_zone: str, end_pos: Position3D) -> Optional[List[Dict]]:
        """
        Find the best flight path between two locations
        
        Args:
            start_zone: Starting zone
            start_pos: Starting position
            end_zone: Ending zone
            end_pos: Ending position
            
        Returns:
            Optional[List[Dict]]: List of navigation actions or None
        """
        # Get nearest flight points
        start_fp = self.get_nearest_flight_point(start_pos, start_zone)
        end_fp = self.get_nearest_flight_point(end_pos, end_zone)
        
        if not start_fp or not end_fp:
            return None
        
        # Check if both flight points are discovered
        if not start_fp.discovered:
            self.logger.warning(f"Starting flight point {start_fp.name} not discovered")
            return None
        
        # Find best path through flight network
        flight_path = self._find_best_flight_path(start_fp, end_fp)
        
        if not flight_path:
            return None
        
        # Create navigation actions
        actions = []
        
        # 1. Navigate to the starting flight point
        actions.append({
            "type": "move",
            "position": start_fp.position.to_tuple(),
            "description": f"Move to flight master at {start_fp.name}"
        })
        
        # 2. Add flight actions for each hop
        for i in range(len(flight_path) - 1):
            source = flight_path[i]
            target = flight_path[i + 1]
            
            actions.append({
                "type": "use_flight_path",
                "source": source.name,
                "destination": target.name,
                "description": f"Take flight from {source.name} to {target.name}"
            })
        
        # 3. Navigate from final flight point to destination
        actions.append({
            "type": "move",
            "position": end_pos.to_tuple(),
            "description": f"Move from {end_fp.name} to destination"
        })
        
        return actions
    
    def _find_best_flight_path(self, start_fp: FlightNode, end_fp: FlightNode) -> Optional[List[FlightNode]]:
        """
        Find the best flight path using Dijkstra's algorithm
        
        Args:
            start_fp: Starting flight point
            end_fp: Ending flight point
            
        Returns:
            Optional[List[FlightNode]]: Best path or None
        """
        # Initialize Dijkstra's algorithm
        distances = {start_fp: 0}
        previous = {}
        unvisited = set(self.flight_nodes.values())
        
        # Main loop
        while unvisited:
            # Find the unvisited node with the smallest distance
            current = None
            current_distance = float('inf')
            
            for node in unvisited:
                if node in distances and distances[node] < current_distance:
                    current = node
                    current_distance = distances[node]
            
            # If we've processed the end node or there are no more reachable nodes
            if current is None or current == end_fp:
                break
            
            # Remove current from unvisited
            unvisited.remove(current)
            
            # Check all connections from current
            for connection in current.get_connections():
                # Skip undiscovered connections
                if not connection.target.discovered:
                    continue
                
                # Calculate new distance
                distance = distances[current] + connection.cost
                
                # If we found a shorter path, update it
                if connection.target not in distances or distance < distances[connection.target]:
                    distances[connection.target] = distance
                    previous[connection.target] = current
        
        # Check if we found a path
        if end_fp not in distances:
            return None
        
        # Reconstruct the path
        path = [end_fp]
        current = end_fp
        
        while current in previous:
            current = previous[current]
            path.append(current)
        
        # Reverse path (start to end)
        path.reverse()
        
        return path
    
    def find_all_flight_paths(self, start_zone: str, start_pos: Position3D) -> List[Dict]:
        """
        Find all available flight paths from a starting location
        
        Args:
            start_zone: Starting zone
            start_pos: Starting position
            
        Returns:
            List[Dict]: List of available flight destinations with details
        """
        # Get nearest flight point
        start_fp = self.get_nearest_flight_point(start_pos, start_zone)
        
        if not start_fp:
            return []
        
        # Check if flight point is discovered
        if not start_fp.discovered:
            return []
        
        # Find all reachable destinations
        destinations = []
        
        # Initialize BFS
        visited = {start_fp}
        queue = [(start_fp, 0, [])]  # (node, cost, path)
        
        while queue:
            current, cost, path = queue.pop(0)
            
            # Add to destinations if not the start
            if current != start_fp:
                destinations.append({
                    "name": current.name,
                    "zone": current.zone,
                    "position": current.position.to_tuple(),
                    "cost": cost,
                    "path": [node.name for node in path + [current]]
                })
            
            # Check all connections
            for connection in current.get_connections():
                if not connection.target.discovered or connection.target in visited:
                    continue
                
                visited.add(connection.target)
                queue.append((
                    connection.target,
                    cost + connection.cost,
                    path + [current]
                ))
        
        return destinations
    
    def estimate_travel_time(self, start_zone: str, start_pos: Position3D,
                          end_zone: str, end_pos: Position3D) -> Dict:
        """
        Estimate travel time for different methods (walking vs flying)
        
        Args:
            start_zone: Starting zone
            start_pos: Starting position
            end_zone: Ending zone
            end_pos: Ending position
            
        Returns:
            Dict: Estimated travel times for different methods
        """
        # Calculate direct distance
        # For cross-zone, this is an approximation
        direct_distance = 0
        
        if start_zone == end_zone:
            # Same zone, direct distance calculation
            direct_distance = math.sqrt(
                (start_pos.x - end_pos.x) ** 2 +
                (start_pos.y - end_pos.y) ** 2 +
                (start_pos.z - end_pos.z) ** 2
            )
        else:
            # Cross-zone, approximate with zone centers
            # This is very rough and would need proper zone data
            direct_distance = 1000  # Default large distance
        
        # Estimate walking time (assume 7 units/sec walking speed)
        walking_time = direct_distance / 7.0  # seconds
        
        # Estimate running time (assume 14 units/sec running speed)
        running_time = direct_distance / 14.0  # seconds
        
        # Estimate mounted time (assume 24 units/sec mounted speed)
        mounted_time = direct_distance / 24.0  # seconds
        
        # Estimate flight time
        flight_path = self.find_flight_path(start_zone, start_pos, end_zone, end_pos)
        
        flight_time = None
        if flight_path:
            # Count flight hops
            hop_count = sum(1 for action in flight_path if action["type"] == "use_flight_path")
            
            # Calculate time to first flight point
            first_fp_time = 0
            if flight_path and flight_path[0]["type"] == "move":
                fp_pos = flight_path[0]["position"]
                fp_distance = math.sqrt(
                    (start_pos.x - fp_pos[0]) ** 2 +
                    (start_pos.y - fp_pos[1]) ** 2 +
                    (start_pos.z - fp_pos[2]) ** 2
                )
                first_fp_time = fp_distance / 14.0  # seconds (running)
            
            # Calculate time from last flight point to destination
            last_fp_time = 0
            if flight_path and flight_path[-1]["type"] == "move":
                fp_pos = flight_path[-2]["position"] if len(flight_path) > 1 else start_pos.to_tuple()
                dest_distance = math.sqrt(
                    (fp_pos[0] - end_pos.x) ** 2 +
                    (fp_pos[1] - end_pos.y) ** 2 +
                    (fp_pos[2] - end_pos.z) ** 2
                )
                last_fp_time = dest_distance / 14.0  # seconds (running)
            
            # Estimate flight time (1 minute minimum per hop, plus distance factor)
            hop_time = hop_count * 60  # seconds (1 minute per hop minimum)
            
            # Total flight time
            flight_time = first_fp_time + hop_time + last_fp_time
        
        return {
            "walking": walking_time,
            "running": running_time,
            "mounted": mounted_time,
            "flight": flight_time,
            "recommended": self._get_recommended_travel_method(
                walking_time, running_time, mounted_time, flight_time)
        }
    
    def _get_recommended_travel_method(self, walking_time, running_time, 
                                     mounted_time, flight_time) -> str:
        """
        Get the recommended travel method based on time estimates
        
        Args:
            walking_time: Estimated walking time
            running_time: Estimated running time
            mounted_time: Estimated mounted time
            flight_time: Estimated flight time
            
        Returns:
            str: Recommended travel method
        """
        # Create a list of methods and times (excluding None values)
        methods = []
        
        if walking_time is not None:
            methods.append(("walking", walking_time))
        
        if running_time is not None:
            methods.append(("running", running_time))
        
        if mounted_time is not None:
            methods.append(("mounted", mounted_time))
        
        if flight_time is not None:
            methods.append(("flight", flight_time))
        
        if not methods:
            return "unknown"
        
        # Sort by time (fastest first)
        methods.sort(key=lambda x: x[1])
        
        # Consider convenience factor - prefer flight for long distances
        # even if slightly slower (loading screens are a trade-off for AFK travel)
        if len(methods) > 1:
            fastest_method, fastest_time = methods[0]
            
            for method, time in methods:
                if method == "flight" and time < fastest_time * 1.25:
                    # If flight is within 25% of the fastest method, prefer flight
                    return "flight"
        
        # Return the fastest method
        return methods[0][0]
    
    def get_zone_transition_flight(self, start_zone: str, end_zone: str) -> Optional[Dict]:
        """
        Find the best flight path for a zone transition
        
        Args:
            start_zone: Starting zone
            end_zone: Ending zone
            
        Returns:
            Optional[Dict]: Best flight path for zone transition or None
        """
        # Check if both zones have flight points
        if start_zone not in self.zone_flight_nodes or end_zone not in self.zone_flight_nodes:
            return None
        
        # Get all discovered flight points in each zone
        start_fps = [fp for fp in self.zone_flight_nodes[start_zone] if fp.discovered]
        end_fps = [fp for fp in self.zone_flight_nodes[end_zone] if fp.discovered]
        
        if not start_fps or not end_fps:
            return None
        
        # Find the best path between any pair of flight points
        best_path = None
        best_cost = float('inf')
        
        for start_fp in start_fps:
            for end_fp in end_fps:
                path = self._find_best_flight_path(start_fp, end_fp)
                
                if path:
                    # Calculate total cost
                    total_cost = 0
                    for i in range(len(path) - 1):
                        connection = path[i].connections.get(path[i+1].name)
                        if connection:
                            total_cost += connection.cost
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path
        
        if best_path:
            return {
                "path": [fp.name for fp in best_path],
                "cost": best_cost,
                "start_zone": start_zone,
                "end_zone": end_zone
            }
        
        return None