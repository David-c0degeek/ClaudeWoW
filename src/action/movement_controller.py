"""
Movement Controller Module

This module handles movement execution in the game:
- Translates navigation actions into game inputs
- Manages movement types (walking, running, flying)
- Handles special movement actions (jumping, swimming)
- Maps 3D movements to 2D inputs
"""

import logging
import math
import time
from typing import Dict, List, Tuple, Any, Optional, Union

from src.action.controller import Controller
from src.perception.screen_reader import GameState
from src.decision.advanced_navigation import Position3D

class MovementController:
    """
    Handles movement execution in the game world
    """
    
    def __init__(self, config: Dict, controller: Controller):
        """
        Initialize the MovementController
        
        Args:
            config: Configuration dictionary
            controller: Game controller for input execution
        """
        self.logger = logging.getLogger("wow_ai.action.movement_controller")
        self.config = config
        self.controller = controller
        
        # Movement settings
        self.movement_speed = config.get("movement_speed", 14.0)  # units per second
        self.rotation_speed = config.get("rotation_speed", 180.0)  # degrees per second
        self.jump_interval = config.get("jump_interval", 0.5)  # seconds
        self.key_press_interval = config.get("key_press_interval", 0.1)  # seconds
        
        # Movement state
        self.current_action = None
        self.action_start_time = 0
        self.last_position = None
        self.last_rotation = 0
        self.is_moving = False
        self.movement_keys = {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "jump": "space",
            "swim_up": "space",
            "swim_down": "x",
            "mount": self.config.get("mount_key", "shift+p")
        }
        
        # Update movement keys from config if provided
        movement_config = config.get("movement_keys", {})
        for key, value in movement_config.items():
            if key in self.movement_keys:
                self.movement_keys[key] = value
        
        self.logger.info("MovementController initialized")
    
    def execute_movement(self, action: Dict, state: GameState) -> bool:
        """
        Execute a movement action
        
        Args:
            action: Movement action to execute
            state: Current game state
            
        Returns:
            bool: True if action complete, False if still in progress
        """
        # Store current action if new
        if self.current_action != action:
            self.current_action = action
            self.action_start_time = time.time()
            self.logger.info(f"Starting new movement action: {action['type']}")
        
        # Handle different action types
        if action["type"] == "move":
            return self._execute_move(action, state)
        elif action["type"] == "jump":
            return self._execute_jump(action, state)
        elif action["type"] == "turn":
            return self._execute_turn(action, state)
        elif action["type"] == "strafe":
            return self._execute_strafe(action, state)
        elif action["type"] == "use_flight_path":
            return self._execute_flight_path(action, state)
        elif action["type"] == "use_portal":
            return self._execute_portal(action, state)
        elif action["type"] == "open_door":
            return self._execute_interact(action, state)
        elif action["type"] == "mount":
            return self._execute_mount(action, state)
        elif action["type"] == "dismount":
            return self._execute_dismount(action, state)
        elif action["type"] == "zone_transition":
            # Zone transitions are passive, just mark as complete
            return True
        else:
            self.logger.warning(f"Unknown movement action type: {action['type']}")
            return True
    
    def _execute_move(self, action: Dict, state: GameState) -> bool:
        """
        Execute a move action
        
        Args:
            action: Move action
            state: Current game state
            
        Returns:
            bool: True if movement complete, False if still in progress
        """
        # Get current position and target
        current_position = None
        if hasattr(state, "player_position"):
            current_position = state.player_position
        
        if not current_position:
            self.logger.warning("Cannot execute move: unknown current position")
            return False
        
        # Get target position
        target_position = action.get("position")
        if not target_position:
            self.logger.warning("Cannot execute move: no target position")
            return True  # Mark as complete to avoid getting stuck
        
        # Calculate distance to target
        distance = self._calculate_distance(current_position, target_position)
        
        # Check if we've reached the target
        if distance < 2.0:
            # Stop movement
            self._stop_movement()
            self.logger.info(f"Reached target position: {target_position}")
            return True
        
        # Calculate direction to target
        direction = self._calculate_direction(current_position, target_position)
        
        # Calculate player rotation
        player_rotation = state.player_rotation if hasattr(state, "player_rotation") else 0
        
        # Calculate angle difference
        angle_diff = self._calculate_angle_difference(direction, player_rotation)
        
        # Turn to face target if needed
        if abs(angle_diff) > 20:
            # Calculate turn direction
            turn_direction = "right" if angle_diff > 0 else "left"
            
            # Turn towards target
            self._turn(turn_direction)
            return False
        
        # Check for obstacles in front
        if hasattr(state, "obstacle_front") and state.obstacle_front:
            # Try to jump over obstacle
            self._jump()
            
            # Continue moving
            self._move_forward()
            return False
        
        # Check if we're already moving
        if not self.is_moving:
            # Start moving forward
            self._move_forward()
        
        # Calculate estimated arrival time
        estimated_time = distance / self.movement_speed
        
        # If we've been moving longer than expected, recalculate
        current_time = time.time()
        time_moving = current_time - self.action_start_time
        
        if time_moving > estimated_time * 1.5:
            # We might be stuck, try jumping
            self._jump()
        
        # Continue moving
        return False
    
    def _execute_jump(self, action: Dict, state: GameState) -> bool:
        """
        Execute a jump action
        
        Args:
            action: Jump action
            state: Current game state
            
        Returns:
            bool: True if jump complete, False if still in progress
        """
        # Perform jump
        self._jump()
        
        # Jumps are instantaneous
        return True
    
    def _execute_turn(self, action: Dict, state: GameState) -> bool:
        """
        Execute a turn action
        
        Args:
            action: Turn action
            state: Current game state
            
        Returns:
            bool: True if turn complete, False if still in progress
        """
        # Get target angle
        target_angle = action.get("angle", 0)
        
        # Get current rotation
        current_rotation = state.player_rotation if hasattr(state, "player_rotation") else 0
        
        # Calculate angle difference
        angle_diff = self._calculate_angle_difference(target_angle, current_rotation)
        
        # Check if we've reached the target angle
        if abs(angle_diff) < 5:
            # Stop turning
            self._stop_turning()
            return True
        
        # Calculate turn direction
        turn_direction = "right" if angle_diff > 0 else "left"
        
        # Start turning
        self._turn(turn_direction)
        
        # Continue turning
        return False
    
    def _execute_strafe(self, action: Dict, state: GameState) -> bool:
        """
        Execute a strafe action
        
        Args:
            action: Strafe action
            state: Current game state
            
        Returns:
            bool: True if strafe complete, False if still in progress
        """
        # Get direction and duration
        direction = action.get("direction", "right")
        duration = action.get("duration", 1.0)
        
        # Check if we've been strafing long enough
        current_time = time.time()
        time_strafing = current_time - self.action_start_time
        
        if time_strafing >= duration:
            # Stop strafing
            self._stop_strafing()
            return True
        
        # Start strafing if not already
        if not self.is_moving:
            self._strafe(direction)
        
        # Continue strafing
        return False
    
    def _execute_flight_path(self, action: Dict, state: GameState) -> bool:
        """
        Execute taking a flight path
        
        Args:
            action: Flight path action
            state: Current game state
            
        Returns:
            bool: True if complete, False if still in progress
        """
        # Get source and destination
        source = action.get("source", "")
        destination = action.get("destination", "")
        
        # Check if we're already on a flight path
        if hasattr(state, "on_flight_path") and state.on_flight_path:
            # Check if we've arrived
            if hasattr(state, "current_subzone"):
                current_zone = state.current_subzone.lower()
                
                # Check if destination name appears in current zone
                if destination.lower() in current_zone:
                    self.logger.info(f"Arrived at flight destination: {destination}")
                    return True
            
            # Still flying
            return False
        
        # Check if we're at a flight master
        if hasattr(state, "nearest_npc_type") and state.nearest_npc_type == "flight_master":
            # Interact with flight master if not already
            self._interact()
            
            # Wait for flight UI
            time.sleep(1.0)
            
            # Need to select destination
            if hasattr(state, "ui_state") and state.ui_state.get("flight_ui_open", False):
                # Try to select destination
                self._select_flight_destination(destination, state)
                
                # Wait for selection
                time.sleep(0.5)
                
                # Confirm flight
                self.controller.press_key("enter")
                
                # Wait for flight to start
                time.sleep(2.0)
                
                return False
        
        # Need to wait for flight to complete
        return False
    
    def _execute_portal(self, action: Dict, state: GameState) -> bool:
        """
        Execute using a portal
        
        Args:
            action: Portal action
            state: Current game state
            
        Returns:
            bool: True if complete, False if still in progress
        """
        # Check if we're near a portal
        if hasattr(state, "nearest_object_type") and state.nearest_object_type == "portal":
            # Interact with portal
            self._interact()
            
            # Portal use is instantaneous, but loading screen takes time
            # Check if zone has changed
            if hasattr(state, "current_zone") and hasattr(state, "previous_zone"):
                if state.current_zone != state.previous_zone:
                    return True
            
            # Wait a bit for loading screen
            time.sleep(2.0)
            return False
        
        # Not near a portal, action failed
        self.logger.warning("Cannot use portal: no portal nearby")
        return True
    
    def _execute_interact(self, action: Dict, state: GameState) -> bool:
        """
        Execute interaction with object
        
        Args:
            action: Interact action
            state: Current game state
            
        Returns:
            bool: True if complete, False if still in progress
        """
        # Perform interaction
        self._interact()
        
        # Interactions are instantaneous
        return True
    
    def _execute_mount(self, action: Dict, state: GameState) -> bool:
        """
        Execute mounting
        
        Args:
            action: Mount action
            state: Current game state
            
        Returns:
            bool: True if complete, False if still in progress
        """
        # Check if already mounted
        if hasattr(state, "mounted") and state.mounted:
            return True
        
        # Mount up
        self.controller.press_key(self.movement_keys["mount"])
        
        # Wait for mount animation
        time.sleep(2.0)
        
        return True
    
    def _execute_dismount(self, action: Dict, state: GameState) -> bool:
        """
        Execute dismounting
        
        Args:
            action: Dismount action
            state: Current game state
            
        Returns:
            bool: True if complete, False if still in progress
        """
        # Check if mounted
        if hasattr(state, "mounted") and state.mounted:
            # Dismount by pressing mount key again
            self.controller.press_key(self.movement_keys["mount"])
            
            # Wait for dismount animation
            time.sleep(1.0)
        
        return True
    
    def _move_forward(self) -> None:
        """Start moving forward"""
        self.controller.press_key(self.movement_keys["forward"], hold=True)
        self.is_moving = True
    
    def _move_backward(self) -> None:
        """Start moving backward"""
        self.controller.press_key(self.movement_keys["backward"], hold=True)
        self.is_moving = True
    
    def _strafe(self, direction: str) -> None:
        """
        Start strafing in a direction
        
        Args:
            direction: Direction to strafe (left, right)
        """
        if direction == "left":
            self.controller.press_key(self.movement_keys["left"], hold=True)
        else:
            self.controller.press_key(self.movement_keys["right"], hold=True)
        
        self.is_moving = True
    
    def _turn(self, direction: str) -> None:
        """
        Start turning in a direction
        
        Args:
            direction: Direction to turn (left, right)
        """
        if direction == "left":
            self.controller.move_mouse(-5, 0, relative=True)
        else:
            self.controller.move_mouse(5, 0, relative=True)
    
    def _jump(self) -> None:
        """Perform a jump"""
        self.controller.press_key(self.movement_keys["jump"])
    
    def _interact(self) -> None:
        """Interact with an object (default right click)"""
        self.controller.right_click()
    
    def _stop_movement(self) -> None:
        """Stop all movement"""
        keys = [
            self.movement_keys["forward"],
            self.movement_keys["backward"],
            self.movement_keys["left"],
            self.movement_keys["right"]
        ]
        
        for key in keys:
            self.controller.release_key(key)
        
        self.is_moving = False
    
    def _stop_turning(self) -> None:
        """Stop turning"""
        # For mouse turning, just stop moving the mouse
        pass
    
    def _stop_strafing(self) -> None:
        """Stop strafing"""
        self.controller.release_key(self.movement_keys["left"])
        self.controller.release_key(self.movement_keys["right"])
        self.is_moving = False
    
    def _select_flight_destination(self, destination: str, state: GameState) -> None:
        """
        Select a flight destination
        
        Args:
            destination: Destination name
            state: Current game state
        """
        # Get available destinations
        destinations = []
        if hasattr(state, "ui_state"):
            destinations = state.ui_state.get("flight_destinations", [])
        
        # Find closest match
        best_match = None
        best_match_score = 0
        
        for i, dest in enumerate(destinations):
            # Calculate simple string match score
            score = self._string_match_score(destination.lower(), dest.lower())
            
            if score > best_match_score:
                best_match_score = score
                best_match = i
        
        if best_match is not None:
            # Select the destination
            # In a real implementation, this would scroll to and click the destination
            # For simplicity, we'll just press down arrow to navigate
            
            for _ in range(best_match):
                self.controller.press_key("down")
                time.sleep(0.1)
    
    def _string_match_score(self, a: str, b: str) -> float:
        """
        Calculate a simple string match score
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            float: Match score (0-1)
        """
        # Check if one string contains the other
        if a in b or b in a:
            return 0.8
        
        # Count matching characters
        matches = sum(1 for c in a if c in b)
        return matches / max(len(a), len(b))
    
    def _calculate_distance(self, pos1: Union[Tuple[float, float], Tuple[float, float, float]],
                         pos2: Union[Tuple[float, float], Tuple[float, float, float]]) -> float:
        """
        Calculate distance between two positions
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            float: Distance
        """
        # Calculate 2D distance
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance_2d = math.sqrt(dx**2 + dy**2)
        
        # Calculate 3D distance if z coordinates available
        if len(pos1) > 2 and len(pos2) > 2:
            dz = pos2[2] - pos1[2]
            return math.sqrt(distance_2d**2 + dz**2)
        
        return distance_2d
    
    def _calculate_direction(self, pos1: Union[Tuple[float, float], Tuple[float, float, float]],
                          pos2: Union[Tuple[float, float], Tuple[float, float, float]]) -> float:
        """
        Calculate direction (angle) from pos1 to pos2
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            float: Direction angle in degrees
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Calculate angle in radians
        angle_rad = math.atan2(dy, dx)
        
        # Convert to degrees (0-360)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def _calculate_angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate the smallest difference between two angles
        
        Args:
            angle1: First angle
            angle2: Second angle
            
        Returns:
            float: Angle difference (-180 to 180)
        """
        # Normalize angles to 0-360
        angle1 = angle1 % 360
        angle2 = angle2 % 360
        
        # Calculate raw difference
        diff = angle1 - angle2
        
        # Normalize to -180 to 180
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        return diff