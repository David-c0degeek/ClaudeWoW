"""
Minimap Analyzer Module

This module analyzes the minimap to extract navigation information.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import math

class MinimapAnalyzer:
    """
    Analyzes the minimap to extract navigation and location information
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MinimapAnalyzer
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.perception.minimap_analyzer")
        self.config = config
        
        # Define colors for different minimap icons
        self.icon_colors = {
            "player": (255, 255, 255),         # White (player arrow)
            "quest_available": (255, 255, 0),   # Yellow (!)
            "quest_complete": (255, 255, 0),    # Yellow (?)
            "vendor": (0, 255, 255),            # Cyan
            "flight_master": (0, 255, 0),       # Green
            "inn": (0, 0, 255),                 # Blue
            "resource_node": (255, 165, 0),     # Orange
            "enemy": (255, 0, 0)                # Red
        }
        
        # Hardcoded minimap radius (will be calculated dynamically in a real implementation)
        self.minimap_radius = 80
        
        self.logger.info("MinimapAnalyzer initialized")
    
    def analyze_minimap(self, screenshot: np.ndarray, minimap_rect: Optional[Tuple[int, int, int, int]]) -> Dict:
        """
        Analyze the minimap to extract navigation information
        
        Args:
            screenshot: The game screenshot
            minimap_rect: Rectangle coordinates of the minimap (x, y, width, height) or None
        
        Returns:
            Dict: Minimap analysis results
        """
        result = {
            "player_position": (0, 0),
            "player_rotation": 0,
            "nodes": [],
            "npcs": [],
            "enemies": [],
            "quest_markers": []
        }
        
        if minimap_rect is None:
            self.logger.warning("Minimap rectangle not provided")
            return result
        
        try:
            x, y, w, h = minimap_rect
            
            # Extract the minimap region
            minimap = screenshot[y:y+h, x:x+w]
            
            # Find the center of the minimap (usually the player's position)
            center_x, center_y = w // 2, h // 2
            
            # Set player position as center of minimap
            result["player_position"] = (center_x, center_y)
            
            # Detect player rotation (arrow direction)
            rotation = self._detect_player_arrow(minimap, (center_x, center_y))
            result["player_rotation"] = rotation
            
            # Detect minimap icons
            icons = self._detect_minimap_icons(minimap)
            
            # Categorize icons based on their type
            for icon in icons:
                icon_type = icon.get("type")
                if icon_type in ["quest_available", "quest_complete"]:
                    result["quest_markers"].append(icon)
                elif icon_type == "enemy":
                    result["enemies"].append(icon)
                elif icon_type in ["vendor", "flight_master", "inn"]:
                    result["npcs"].append(icon)
                elif icon_type == "resource_node":
                    result["nodes"].append(icon)
            
        except Exception as e:
            self.logger.error(f"Error analyzing minimap: {e}")
        
        return result
    
    def _detect_player_arrow(self, minimap: np.ndarray, center: Tuple[int, int]) -> float:
        """
        Detect the player's arrow direction on the minimap
        
        Args:
            minimap: The minimap image
            center: Center coordinates of the minimap (x, y)
        
        Returns:
            float: Player rotation in degrees (0-360)
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(minimap, cv2.COLOR_RGB2HSV)
            
            # Define range for white color (player arrow)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            # Create mask for white color
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the player arrow contour
            player_contour = None
            for contour in contours:
                # Get the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Check if the contour is near the center
                    dist_from_center = math.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                    if dist_from_center < 20:  # Assume player arrow is within 20 pixels of center
                        player_contour = contour
                        break
            
            if player_contour is not None:
                # Find orientation using principal component analysis
                pca_output = cv2.PCACompute2(np.float32(player_contour).reshape(-1, 2), np.mean(np.float32(player_contour).reshape(-1, 2), 0).reshape(1, -1))
                eigenvectors = pca_output[1]
                
                # The first eigenvector points in the direction of greatest variance
                direction = eigenvectors[0]
                
                # Calculate angle in degrees
                angle = math.atan2(direction[1], direction[0]) * 180 / math.pi
                
                # Adjust angle to be between 0 and 360 degrees
                angle = (angle + 360) % 360
                
                return angle
            
        except Exception as e:
            self.logger.error(f"Error detecting player arrow: {e}")
        
        # Default to 0 degrees if detection fails
        return 0.0
    
    def _detect_minimap_icons(self, minimap: np.ndarray) -> List[Dict]:
        """
        Detect icons on the minimap
        
        Args:
            minimap: The minimap image
        
        Returns:
            List[Dict]: Detected minimap icons
        """
        icons = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(minimap, cv2.COLOR_RGB2HSV)
            
            # Process each icon color
            for icon_type, color in self.icon_colors.items():
                if icon_type == "player":
                    continue  # Player already processed separately
                
                # Convert RGB to HSV
                hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
                
                # Create a color range for the mask
                lower_bound = np.array([hsv_color[0] - 10, 100, 100])
                upper_bound = np.array([hsv_color[0] + 10, 255, 255])
                
                # Create mask for the color
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Process each contour as a potential icon
                for i, contour in enumerate(contours):
                    # Filter out very small contours
                    if cv2.contourArea(contour) < 10:
                        continue
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Get the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w // 2, y + h // 2
                    
                    # Calculate distance from center of minimap
                    center_x, center_y = minimap.shape[1] // 2, minimap.shape[0] // 2
                    distance = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    # Calculate angle from center
                    angle = math.atan2(cy - center_y, cx - center_x) * 180 / math.pi
                    angle = (angle + 360) % 360
                    
                    # Create icon entry
                    icon = {
                        "type": icon_type,
                        "position": (cx, cy),
                        "distance": distance,
                        "angle": angle,
                        "rect": (x, y, w, h)
                    }
                    
                    icons.append(icon)
            
        except Exception as e:
            self.logger.error(f"Error detecting minimap icons: {e}")
        
        return icons
    
    def calculate_world_position(self, minimap_position: Tuple[int, int], 
                                player_world_pos: Tuple[float, float],
                                player_rotation: float) -> Tuple[float, float]:
        """
        Calculate world position from minimap position
        
        Args:
            minimap_position: Position on the minimap (x, y)
            player_world_pos: Player's current world position
            player_rotation: Player's rotation in degrees
        
        Returns:
            Tuple[float, float]: Calculated world position (x, y)
        """
        # This is a simplified implementation
        # In a real implementation, we would need to:
        # 1. Calculate the relative position from the center
        # 2. Apply rotation based on player orientation
        # 3. Scale based on minimap zoom level
        # 4. Add to player's current world position
        
        try:
            # Get minimap center
            center_x, center_y = self.minimap_radius, self.minimap_radius
            
            # Calculate relative position from center
            rel_x = minimap_position[0] - center_x
            rel_y = minimap_position[1] - center_y
            
            # Convert player rotation to radians
            rotation_rad = math.radians(player_rotation)
            
            # Apply rotation
            rotated_x = rel_x * math.cos(rotation_rad) - rel_y * math.sin(rotation_rad)
            rotated_y = rel_x * math.sin(rotation_rad) + rel_y * math.cos(rotation_rad)
            
            # Scale based on minimap radius to world units
            # This scaling factor would be calibrated in a real implementation
            scale_factor = 1.0
            world_rel_x = rotated_x * scale_factor
            world_rel_y = rotated_y * scale_factor
            
            # Add to player's world position
            world_x = player_world_pos[0] + world_rel_x
            world_y = player_world_pos[1] + world_rel_y
            
            return (world_x, world_y)
            
        except Exception as e:
            self.logger.error(f"Error calculating world position: {e}")
            return player_world_pos  # Return player position if calculation fails