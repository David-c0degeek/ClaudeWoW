"""
Navigation Manager Module

This module handles navigation, pathfinding, and exploration.
"""

import logging
import random
import math
from typing import Dict, List, Tuple, Any, Optional
import time
import heapq

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge

class NavigationManager:
    """
    Manages navigation, pathfinding, and exploration
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the NavigationManager
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.navigation_manager")
        self.config = config
        self.knowledge = knowledge
        
        # Navigation settings
        self.path_recalculation_interval = config.get("path_recalculation_interval", 5.0)  # seconds
        self.stuck_detection_threshold = config.get("stuck_detection_threshold", 3.0)  # seconds
        self.waypoint_reached_distance = config.get("waypoint_reached_distance", 5.0)  # game units
        
        # Current navigation state
        self.current_path = []
        self.current_waypoint_index = 0
        self.last_path_calculation_time = 0
        self.last_position = None
        self.last_position_time = 0
        self.is_stuck = False
        self.stuck_time = 0
        
        # Exploration state
        self.explored_areas = set()
        self.exploration_targets = []
        self.last_exploration_target = None
        
        self.logger.info("NavigationManager initialized")
    
    def generate_navigation_plan(self, state: GameState, destination: Tuple[float, float]) -> List[Dict]:
        """
        Generate a navigation plan to reach a specific destination
        
        Args:
            state: Current game state
            destination: Destination coordinates (x, y)
        
        Returns:
            List[Tuple[float, float]]: List of waypoints
        """
        # Get current position
        if hasattr(state, "player_position"):
            start = state.player_position
        else:
            self.logger.warning("Player position not found in game state")
            # Return basic movement toward destination
            return [{"type": "move", "position": destination, "description": "Move to destination (fallback)"}]
            
        end = destination
        
        # Check if we can query the knowledge base for a predefined path
        zone = state.current_zone if hasattr(state, "current_zone") else ""
        knowledge_path = self.knowledge.get_path(start, end, zone)
        
        if knowledge_path:
            return knowledge_path
        
        # If no predefined path, use A* pathfinding
        return self._find_path_astar(start, end, state)
    
    def _find_path_astar(self, start: Tuple[float, float], end: Tuple[float, float], 
                       state: GameState) -> List[Tuple[float, float]]:
        """
        Find a path using A* algorithm
        
        Args:
            start: Start position (x, y)
            end: End position (x, y)
            state: Current game state
        
        Returns:
            List[Tuple[float, float]]: List of waypoints
        """
        # This is a simplified A* implementation
        # In a real implementation, we would use a proper navmesh
        
        # Define a simple grid for pathfinding
        # In a real implementation, this would come from the game's map data
        grid_size = 20
        grid_resolution = 10.0  # game units per grid cell
        
        # Initialize data structures
        open_set = []  # Priority queue
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start], start))
        
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if self._calculate_distance(current, end) < grid_resolution:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                path.reverse()
                
                # Optimize path by removing unnecessary waypoints
                optimized_path = self._optimize_path(path)
                
                return optimized_path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Generate neighbors
            neighbors = self._get_neighbors(current, grid_resolution, state)
            
            for neighbor in neighbors:
                # Skip if neighbor in closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g score
                tentative_g = g_score[current] + self._calculate_distance(current, neighbor)
                
                # Check if this path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, end)
                    
                    # Add to open set if not already there
                    if neighbor not in [n for _, n in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If we get here, no path was found
        self.logger.warning("A* pathfinding failed to find a path")
        
        # Return a straight line path as fallback
        return [start, end]
    
    def _get_neighbors(self, position: Tuple[float, float], grid_resolution: float, 
                     state: GameState) -> List[Tuple[float, float]]:
        """
        Get valid neighboring positions
        
        Args:
            position: Current position (x, y)
            grid_resolution: Distance between grid points
            state: Current game state
        
        Returns:
            List[Tuple[float, float]]: List of neighbor positions
        """
        x, y = position
        
        # Generate 8 neighboring positions (cardinal + diagonal directions)
        directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # Cardinal
            (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonal
        ]
        
        neighbors = []
        
        for dx, dy in directions:
            neighbor = (x + dx * grid_resolution, y + dy * grid_resolution)
            
            # Check if the neighbor is valid (not obstructed)
            if self._is_position_valid(neighbor, state):
                neighbors.append(neighbor)
        
        return neighbors
    
    def _is_position_valid(self, position: Tuple[float, float], state: GameState) -> bool:
        """
        Check if a position is valid for navigation
        
        Args:
            position: Position to check (x, y)
            state: Current game state
        
        Returns:
            bool: True if the position is valid
        """
        # In a real implementation, this would check collision with terrain, etc.
        # For this simplified implementation, just check if position is within bounds
        
        # Get minimap data if available
        minimap_data = state.minimap_data if hasattr(state, "minimap_data") else {}
        
        # Check if position is within minimap bounds
        # This is a placeholder implementation
        return True
    
    def _heuristic(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """
        Heuristic function for A* pathfinding (Euclidean distance)
        
        Args:
            a: First position (x, y)
            b: Second position (x, y)
        
        Returns:
            float: Estimated distance between positions
        """
        return self._calculate_distance(a, b)
    
    def _optimize_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Optimize a path by removing unnecessary waypoints
        
        Args:
            path: Original path
        
        Returns:
            List[Tuple[float, float]]: Optimized path
        """
        if len(path) <= 2:
            return path
        
        optimized = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = optimized[-1]
            current = path[i]
            next_pos = path[i + 1]
            
            # Calculate vectors
            v1 = (current[0] - prev[0], current[1] - prev[1])
            v2 = (next_pos[0] - current[0], next_pos[1] - current[1])
            
            # Normalize vectors
            len_v1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
            len_v2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
            
            if len_v1 > 0 and len_v2 > 0:
                v1_norm = (v1[0] / len_v1, v1[1] / len_v1)
                v2_norm = (v2[0] / len_v2, v2[1] / len_v2)
                
                # Calculate dot product to get cosine of angle
                dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
                
                # If directions are different (angle > threshold), keep the waypoint
                if dot_product < 0.9:  # Approximately 25 degrees
                    optimized.append(current)
            else:
                optimized.append(current)
        
        optimized.append(path[-1])
        return optimized
    
    def _is_path_invalid(self, current_position: Tuple[float, float], 
                       destination: Tuple[float, float]) -> bool:
        """
        Check if the current path is invalid and needs to be recalculated
        
        Args:
            current_position: Current player position
            destination: Target destination
        
        Returns:
            bool: True if path is invalid
        """
        # Check if we have a path
        if not self.current_path:
            return True
        
        # Check if destination has changed
        if self.current_path[-1] != destination:
            return True
        
        # Check if we're off the path
        if self.current_waypoint_index < len(self.current_path):
            current_waypoint = self.current_path[self.current_waypoint_index]
            distance = self._calculate_distance(current_position, current_waypoint)
            
            # If we're too far from the current waypoint, path might be invalid
            if distance > self.waypoint_reached_distance * 3:
                return True
        
        return False
    
    def _update_stuck_detection(self, current_position: Tuple[float, float], current_time: float) -> None:
        """
        Update stuck detection state
        
        Args:
            current_position: Current player position
            current_time: Current time
        """
        # Check if we have previous position data
        if not self.last_position or not self.last_position_time:
            return
        
        # Calculate time since last position update
        time_since_last = current_time - self.last_position_time
        
        # If it's been a while, check if we've moved
        if time_since_last > 1.0:
            distance_moved = self._calculate_distance(current_position, self.last_position)
            
            # If we haven't moved significantly, might be stuck
            if distance_moved < 0.5:  # Very little movement
                if not self.is_stuck:
                    self.is_stuck = True
                    self.stuck_time = current_time
                
                # Check if we've been stuck too long
                stuck_duration = current_time - self.stuck_time
                if stuck_duration > self.stuck_detection_threshold:
                    self.logger.warning(f"Detected stuck condition for {stuck_duration:.1f} seconds")
            else:
                # We've moved, no longer stuck
                self.is_stuck = False
                self.stuck_time = 0
    
    def _generate_unstuck_action(self) -> Dict:
        """
        Generate an action to get unstuck
        
        Returns:
            Dict: Unstuck action
        """
        # Choose a random unstuck strategy
        strategies = [
            {"type": "jump", "description": "Jump to get unstuck"},
            {"type": "strafe", "direction": "right", "duration": 1.0, "description": "Strafe right to get unstuck"},
            {"type": "strafe", "direction": "left", "duration": 1.0, "description": "Strafe left to get unstuck"},
            {"type": "move", "direction": "backward", "duration": 2.0, "description": "Move backward to get unstuck"},
            {"type": "turn", "angle": random.uniform(-180, 180), "description": "Turn around to get unstuck"}
        ]
        
        return random.choice(strategies)
    
    def _find_exploration_target(self, state: GameState) -> Optional[Dict]:
        """
        Find a suitable exploration target
        
        Args:
            state: Current game state
        
        Returns:
            Optional[Dict]: Exploration target info
        """
        # Check minimap for interesting points
        if hasattr(state, "minimap_data") and state.minimap_data:
            # Check for quest markers
            quest_markers = state.minimap_data.get("quest_markers", [])
            if quest_markers:
                return {
                    "type": "quest_marker",
                    "position": quest_markers[0].get("position"),
                    "interactive": True,
                    "id": quest_markers[0].get("id", "quest_marker")
                }
            
            # Check for resource nodes
            nodes = state.minimap_data.get("nodes", [])
            if nodes:
                return {
                    "type": "resource_node",
                    "position": nodes[0].get("position"),
                    "interactive": True,
                    "id": nodes[0].get("id", "resource_node")
                }
            
            # Check for NPCs
            npcs = state.minimap_data.get("npcs", [])
            if npcs:
                return {
                    "type": "npc",
                    "position": npcs[0].get("position"),
                    "interactive": True,
                    "id": npcs[0].get("id", "npc")
                }
        
        # Check for unexplored areas
        if hasattr(state, "current_zone"):
            zone = state.current_zone
            unexplored = self.knowledge.get_unexplored_areas(zone, self.explored_areas)
            
            if unexplored:
                return {
                    "type": "unexplored_area",
                    "position": unexplored[0].get("position"),
                    "interactive": False,
                    "id": unexplored[0].get("id", "area")
                }
        
        return None
    
    def _generate_random_exploration(self, state: GameState) -> List[Dict]:
        """
        Generate random exploration movement
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Exploration actions
        """
        current_position = state.player_position if hasattr(state, "player_position") else (0, 0)
        
        # Generate a random direction
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(50, 150)  # Random distance
        
        # Calculate target position
        target_x = current_position[0] + distance * math.cos(angle)
        target_y = current_position[1] + distance * math.sin(angle)
        
        return [{
            "type": "move",
            "position": (target_x, target_y),
            "description": "Explore in random direction"
        }]
    
    def _get_current_area(self, state: GameState) -> Optional[str]:
        """
        Get the current area identifier
        
        Args:
            state: Current game state
        
        Returns:
            Optional[str]: Area identifier
        """
        zone = state.current_zone if hasattr(state, "current_zone") else ""
        subzone = state.current_subzone if hasattr(state, "current_subzone") else ""
        
        if zone and subzone:
            return f"{zone}:{subzone}"
        elif zone:
            return zone
        else:
            return None
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate distance between two positions
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
        
        Returns:
            float: Distance between positions
        """
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)