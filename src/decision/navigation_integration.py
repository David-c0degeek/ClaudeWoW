"""
Navigation Integration Module

This module integrates the advanced navigation system with the rest of the AI:
- Coordinates between basic and advanced navigation
- Handles navigation state management
- Provides a unified interface for the agent
"""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.navigation_manager import NavigationManager
from src.decision.advanced_navigation import AdvancedNavigationManager, Position3D
from src.decision.terrain_analyzer import TerrainAnalyzer
from src.decision.flight_path_manager import FlightPathManager
from src.decision.dungeon_navigator import DungeonNavigator

class NavigationIntegrator:
    """
    Integrates all navigation systems and provides a unified interface
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the NavigationIntegrator
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.navigation_integration")
        self.config = config
        self.knowledge = knowledge
        
        # Initialize navigation systems
        self.basic_navigation = NavigationManager(config, knowledge)
        self.terrain_analyzer = TerrainAnalyzer(config)
        self.advanced_navigation = AdvancedNavigationManager(config, knowledge, self.basic_navigation)
        self.flight_path_manager = FlightPathManager(config, knowledge)
        self.dungeon_navigator = DungeonNavigator(config, knowledge)
        
        # Navigation state
        self.current_path = []
        self.current_destination = None
        self.current_destination_zone = None
        self.current_navigation_mode = "basic"  # basic, advanced, flight, dungeon
        self.path_start_time = 0
        self.last_recalculation_time = 0
        
        # Navigation settings
        self.path_recalculation_interval = config.get("path_recalculation_interval", 30.0)  # seconds
        self.stuck_detection_threshold = config.get("stuck_detection_threshold", 10.0)  # seconds
        self.auto_mode_selection = config.get("auto_navigation_mode_selection", True)
        
        self.logger.info("NavigationIntegrator initialized")
    
    def navigate_to(self, state: GameState, destination: Union[Tuple[float, float], Tuple[float, float, float]],
                  destination_zone: str = None, mode: str = "auto") -> List[Dict]:
        """
        Navigate to a destination
        
        Args:
            state: Current game state
            destination: Destination coordinates (x, y) or (x, y, z)
            destination_zone: Destination zone (None for current zone)
            mode: Navigation mode (auto, basic, advanced, flight, dungeon)
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Update current destination
        self.current_destination = destination
        self.current_destination_zone = destination_zone or (
            state.current_zone if hasattr(state, "current_zone") else None
        )
        
        # Record navigation start time
        self.path_start_time = time.time()
        self.last_recalculation_time = time.time()
        
        # If auto mode is selected, determine the best navigation system
        if mode == "auto" and self.auto_mode_selection:
            mode = self._select_navigation_mode(state, destination, destination_zone)
        
        self.current_navigation_mode = mode
        
        # Use the selected navigation system
        if mode == "dungeon" or (mode == "auto" and self.dungeon_navigator.is_in_dungeon(state)):
            self.logger.info("Using dungeon navigation")
            return self._navigate_dungeon(state, destination, destination_zone)
        elif mode == "flight" or (mode == "auto" and self._should_use_flight_path(state, destination, destination_zone)):
            self.logger.info("Using flight path navigation")
            return self._navigate_flight(state, destination, destination_zone)
        elif mode == "advanced":
            self.logger.info("Using advanced navigation")
            return self._navigate_advanced(state, destination, destination_zone)
        else:
            self.logger.info("Using basic navigation")
            return self._navigate_basic(state, destination)
    
    def update_navigation(self, state: GameState) -> List[Dict]:
        """
        Update navigation based on current state
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Updated navigation actions
        """
        # Check if we have a destination
        if not self.current_destination:
            return []
        
        # Check if we've reached the destination
        if self._has_reached_destination(state):
            self.logger.info("Destination reached")
            self.current_path = []
            self.current_destination = None
            self.current_destination_zone = None
            return []
        
        # Check if we're stuck
        if self._is_stuck(state):
            self.logger.info("Detected stuck condition, recalculating path")
            return self.navigate_to(state, self.current_destination, self.current_destination_zone, 
                               self.current_navigation_mode)
        
        # Check if we need to recalculate the path
        current_time = time.time()
        if current_time - self.last_recalculation_time > self.path_recalculation_interval:
            self.logger.info("Recalculating path (interval)")
            self.last_recalculation_time = current_time
            return self.navigate_to(state, self.current_destination, self.current_destination_zone, 
                               self.current_navigation_mode)
        
        # Check for new obstacles
        if hasattr(state, "obstacles") and state.obstacles and self.current_navigation_mode == "advanced":
            # Detect and avoid obstacles
            if hasattr(state, "player_position"):
                current_position = self._get_position_3d(state.player_position)
                
                # Convert current path to Position3D objects
                path_3d = []
                for waypoint in self.current_path:
                    if isinstance(waypoint, dict) and "position" in waypoint:
                        pos = waypoint["position"]
                        path_3d.append(Position3D(pos[0], pos[1], pos[2] if len(pos) > 2 else 0))
                
                if path_3d:
                    # Update path to avoid obstacles
                    updated_path = self.advanced_navigation.detect_and_avoid_obstacles(state, path_3d)
                    
                    if updated_path != path_3d:
                        self.logger.info("Updating path to avoid obstacles")
                        
                        # Convert back to actions
                        return self.advanced_navigation._path_to_actions(updated_path, 
                                                                    state.current_zone if hasattr(state, "current_zone") else "")
        
        # Continue with current path
        return self.current_path
    
    def _navigate_basic(self, state: GameState, destination: Tuple[float, float]) -> List[Dict]:
        """
        Navigate using basic navigation
        
        Args:
            state: Current game state
            destination: Destination coordinates (x, y)
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Ensure destination is 2D
        dest_2d = destination[:2]
        
        actions = self.basic_navigation.generate_navigation_plan(state, dest_2d)
        self.current_path = actions
        return actions
    
    def _navigate_advanced(self, state: GameState, 
                        destination: Union[Tuple[float, float], Tuple[float, float, float]],
                        destination_zone: str = None) -> List[Dict]:
        """
        Navigate using advanced navigation
        
        Args:
            state: Current game state
            destination: Destination coordinates (x, y) or (x, y, z)
            destination_zone: Destination zone
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Ensure destination has z-coordinate
        if len(destination) == 2:
            dest_3d = (destination[0], destination[1], 0)
        else:
            dest_3d = destination
        
        actions = self.advanced_navigation.navigate_to(state, dest_3d, destination_zone)
        self.current_path = actions
        return actions
    
    def _navigate_flight(self, state: GameState,
                      destination: Union[Tuple[float, float], Tuple[float, float, float]],
                      destination_zone: str = None) -> List[Dict]:
        """
        Navigate using flight paths
        
        Args:
            state: Current game state
            destination: Destination coordinates
            destination_zone: Destination zone
            
        Returns:
            List[Dict]: Navigation actions
        """
        current_zone = state.current_zone if hasattr(state, "current_zone") else ""
        
        if not current_zone or not destination_zone:
            # Fall back to advanced navigation
            return self._navigate_advanced(state, destination, destination_zone)
        
        # Get current position
        if hasattr(state, "player_position"):
            if len(state.player_position) >= 3:
                current_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    state.player_position[2]
                )
            else:
                current_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    0
                )
        else:
            # Fall back to advanced navigation
            return self._navigate_advanced(state, destination, destination_zone)
        
        # Get destination position
        if len(destination) >= 3:
            dest_pos = Position3D(destination[0], destination[1], destination[2])
        else:
            dest_pos = Position3D(destination[0], destination[1], 0)
        
        # Find flight path
        actions = self.flight_path_manager.find_flight_path(
            current_zone, current_pos, 
            destination_zone, dest_pos
        )
        
        if not actions:
            # Fall back to advanced navigation
            return self._navigate_advanced(state, destination, destination_zone)
        
        self.current_path = actions
        return actions
    
    def _navigate_dungeon(self, state: GameState,
                       destination: Union[Tuple[float, float], Tuple[float, float, float]],
                       destination_zone: str = None) -> List[Dict]:
        """
        Navigate in a dungeon
        
        Args:
            state: Current game state
            destination: Destination coordinates
            destination_zone: Destination zone
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Detect current dungeon
        current_dungeon = self.dungeon_navigator.detect_current_dungeon(state)
        
        if not current_dungeon:
            # Fall back to advanced navigation
            return self._navigate_advanced(state, destination, destination_zone)
        
        # Check if destination is a boss name
        if isinstance(destination, str):
            # Navigate to boss
            actions = self.dungeon_navigator.navigate_to_boss(state, destination)
            
            if not actions:
                # Fall back to advanced navigation
                return self._navigate_advanced(state, destination, destination_zone)
            
            self.current_path = actions
            return actions
        
        # Otherwise, treat as coordinates
        # Ensure coordinates have z value
        if len(destination) == 2:
            dest_3d = (destination[0], destination[1], 0)
        else:
            dest_3d = destination
        
        # Use advanced navigation within the dungeon
        actions = self.advanced_navigation.navigate_to(state, dest_3d, destination_zone)
        self.current_path = actions
        return actions
    
    def _select_navigation_mode(self, state: GameState, 
                             destination: Union[Tuple[float, float], Tuple[float, float, float]],
                             destination_zone: str = None) -> str:
        """
        Select the best navigation mode for the current situation
        
        Args:
            state: Current game state
            destination: Destination coordinates
            destination_zone: Destination zone
            
        Returns:
            str: Selected navigation mode
        """
        # Check if in dungeon
        if self.dungeon_navigator.is_in_dungeon(state):
            return "dungeon"
        
        # Check if destination is in another zone
        current_zone = state.current_zone if hasattr(state, "current_zone") else None
        
        if destination_zone and current_zone and destination_zone != current_zone:
            # Check if flight paths are available
            if self._should_use_flight_path(state, destination, destination_zone):
                return "flight"
            return "advanced"
        
        # For same zone navigation, check distance
        if hasattr(state, "player_position") and destination:
            current_pos = state.player_position
            
            # Calculate distance
            dx = destination[0] - current_pos[0]
            dy = destination[1] - current_pos[1]
            distance = (dx**2 + dy**2)**0.5
            
            # Use advanced for longer distances or if elevation data is available
            if distance > 200 or len(destination) > 2 or len(current_pos) > 2:
                return "advanced"
        
        # Default to basic navigation
        return "basic"
    
    def _should_use_flight_path(self, state: GameState,
                             destination: Union[Tuple[float, float], Tuple[float, float, float]],
                             destination_zone: str = None) -> bool:
        """
        Determine if flight paths should be used
        
        Args:
            state: Current game state
            destination: Destination coordinates
            destination_zone: Destination zone
            
        Returns:
            bool: True if flight paths should be used
        """
        current_zone = state.current_zone if hasattr(state, "current_zone") else None
        
        if not current_zone or not destination_zone or current_zone == destination_zone:
            return False
        
        # Get current position
        if hasattr(state, "player_position"):
            if len(state.player_position) >= 3:
                current_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    state.player_position[2]
                )
            else:
                current_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    0
                )
        else:
            return False
        
        # Get destination position
        if len(destination) >= 3:
            dest_pos = Position3D(destination[0], destination[1], destination[2])
        else:
            dest_pos = Position3D(destination[0], destination[1], 0)
        
        # Estimate travel times
        travel_times = self.flight_path_manager.estimate_travel_time(
            current_zone, current_pos,
            destination_zone, dest_pos
        )
        
        # Use flight if it's the recommended method
        return travel_times.get("recommended") == "flight"
    
    def _has_reached_destination(self, state: GameState) -> bool:
        """
        Check if the destination has been reached
        
        Args:
            state: Current game state
            
        Returns:
            bool: True if destination reached
        """
        if not self.current_destination:
            return True
        
        if not hasattr(state, "player_position"):
            return False
        
        # Calculate distance to destination
        current_pos = state.player_position
        destination = self.current_destination
        
        dx = destination[0] - current_pos[0]
        dy = destination[1] - current_pos[1]
        distance_2d = (dx**2 + dy**2)**0.5
        
        # Check 3D distance if z coordinates are available
        distance_3d = distance_2d
        if len(destination) > 2 and len(current_pos) > 2:
            dz = destination[2] - current_pos[2]
            distance_3d = (distance_2d**2 + dz**2)**0.5
        
        # Also check zone if applicable
        zone_match = True
        if self.current_destination_zone:
            current_zone = state.current_zone if hasattr(state, "current_zone") else None
            zone_match = current_zone == self.current_destination_zone
        
        # Consider reached if within 5 units and in the right zone
        return distance_2d < 5.0 and zone_match
    
    def _is_stuck(self, state: GameState) -> bool:
        """
        Check if the player is stuck
        
        Args:
            state: Current game state
            
        Returns:
            bool: True if stuck
        """
        if not hasattr(state, "player_position") or not hasattr(state, "last_position"):
            return False
        
        # Calculate distance moved
        current_pos = state.player_position
        last_pos = state.last_position
        
        if not last_pos:
            return False
        
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        distance_moved = (dx**2 + dy**2)**0.5
        
        # Calculate time since starting navigation
        time_navigating = time.time() - self.path_start_time
        
        # Check if we've moved very little over a significant time
        if distance_moved < 1.0 and time_navigating > self.stuck_detection_threshold:
            # Reset the path start time to avoid continuous recalculations
            self.path_start_time = time.time()
            return True
        
        return False
    
    def _get_position_3d(self, position: Union[Tuple[float, float], Tuple[float, float, float]]) -> Position3D:
        """
        Convert a position tuple to a Position3D object
        
        Args:
            position: Position tuple
            
        Returns:
            Position3D: 3D position object
        """
        if len(position) >= 3:
            return Position3D(position[0], position[1], position[2])
        else:
            return Position3D(position[0], position[1], 0)
    
    def navigate_to_instance(self, state: GameState, instance_name: str) -> List[Dict]:
        """
        Navigate to a dungeon or raid instance
        
        Args:
            state: Current game state
            instance_name: Name of the instance
            
        Returns:
            List[Dict]: Navigation actions
        """
        actions = self.advanced_navigation.navigate_to_instance(state, instance_name)
        self.current_path = actions
        self.current_navigation_mode = "advanced"
        return actions
    
    def get_flight_paths(self, state: GameState) -> List[Dict]:
        """
        Get all available flight paths from current location
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Available flight destinations
        """
        current_zone = state.current_zone if hasattr(state, "current_zone") else None
        
        if not current_zone:
            return []
        
        # Get current position
        if hasattr(state, "player_position"):
            if len(state.player_position) >= 3:
                current_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    state.player_position[2]
                )
            else:
                current_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    0
                )
        else:
            return []
        
        return self.flight_path_manager.find_all_flight_paths(current_zone, current_pos)
    
    def analyze_terrain(self, state: GameState, radius: float = 50.0) -> Dict:
        """
        Analyze terrain around player
        
        Args:
            state: Current game state
            radius: Radius to analyze
            
        Returns:
            Dict: Terrain analysis results
        """
        # Get player position
        if hasattr(state, "player_position"):
            if len(state.player_position) >= 3:
                player_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    state.player_position[2]
                )
            else:
                player_pos = Position3D(
                    state.player_position[0],
                    state.player_position[1],
                    0
                )
        else:
            return {}
        
        # Analyze terrain
        terrain_samples = self.terrain_analyzer.analyze_terrain(state, player_pos, radius)
        
        # Convert to simpler format for return
        result = {
            "player_position": player_pos.to_tuple(),
            "terrain_samples": [],
            "obstacles": [],
            "jump_paths": []
        }
        
        # Add terrain samples
        for grid_pos, sample in terrain_samples.items():
            result["terrain_samples"].append({
                "position": sample.position.to_tuple(),
                "type": sample.terrain_type,
                "walkable": sample.is_walkable(),
                "cost": sample.get_movement_cost()
            })
        
        # Add obstacles
        if hasattr(state, "obstacles") and state.obstacles:
            for obs in state.obstacles:
                if isinstance(obs, tuple):
                    result["obstacles"].append({
                        "position": obs,
                        "radius": 2.0  # Default radius
                    })
                elif isinstance(obs, dict):
                    result["obstacles"].append(obs)
        
        # Add jump paths
        jump_paths = self.terrain_analyzer.identify_jump_paths(terrain_samples)
        for start, end in jump_paths:
            result["jump_paths"].append({
                "start": start.to_tuple(),
                "end": end.to_tuple()
            })
        
        return result
    
    def get_dungeon_info(self, dungeon_name: str) -> Optional[Dict]:
        """
        Get information about a dungeon
        
        Args:
            dungeon_name: Name of the dungeon
            
        Returns:
            Optional[Dict]: Dungeon information or None
        """
        return self.dungeon_navigator.get_dungeon_overview(dungeon_name)