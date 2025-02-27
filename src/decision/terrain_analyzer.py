"""
Terrain Analyzer Module

This module provides terrain analysis capabilities:
- Terrain classification from visual data
- Obstacle detection and avoidance
- Slope analysis and traversability assessment
- Jump/fall potential detection
"""

import logging
import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Any, Optional, Set, Union

from src.perception.screen_reader import GameState
from src.decision.advanced_navigation import Position3D, TerrainNode, TerrainType

class TerrainSample:
    """Represents a terrain sample at a specific location"""
    
    def __init__(self, position: Position3D, terrain_type: float = TerrainType.NORMAL,
                color: Tuple[int, int, int] = None, slope: float = 0.0):
        """
        Initialize a terrain sample
        
        Args:
            position: 3D position of the sample
            terrain_type: Type of terrain (affects movement cost)
            color: RGB color of terrain (for visual analysis)
            slope: Slope angle in degrees
        """
        self.position = position
        self.terrain_type = terrain_type
        self.color = color or (128, 128, 128)  # Default gray
        self.slope = slope
        self.metadata = {}
    
    def is_walkable(self) -> bool:
        """Check if this terrain is walkable"""
        return (self.terrain_type != TerrainType.UNWALKABLE and 
                self.slope < 45.0)  # 45 degrees is too steep
    
    def get_movement_cost(self) -> float:
        """Get movement cost considering terrain type and slope"""
        # Base cost from terrain type
        base_cost = self.terrain_type
        
        # Additional cost from slope (steeper = harder)
        slope_factor = 1.0 + (self.slope / 45.0) ** 2
        
        return base_cost * slope_factor

class TerrainAnalyzer:
    """
    Analyzes terrain for navigation purposes
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TerrainAnalyzer
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.decision.terrain_analyzer")
        self.config = config
        
        # Initialize terrain color classification
        self.terrain_colors = {
            TerrainType.NORMAL: [(120, 120, 120), (100, 100, 100), (140, 140, 140)],  # Gray (roads, paths)
            TerrainType.GRASS: [(60, 140, 60), (80, 160, 80), (40, 120, 40)],  # Green (grass)
            TerrainType.WATER_SHALLOW: [(180, 180, 255), (160, 160, 255)],  # Light blue (shallow water)
            TerrainType.WATER_DEEP: [(0, 0, 255), (0, 0, 200)],  # Deep blue (deep water)
            TerrainType.MOUNTAIN: [(160, 120, 80), (180, 140, 100)],  # Brown (mountains)
            TerrainType.CLIFF: [(80, 60, 40), (60, 40, 20)],  # Dark brown (cliffs)
            TerrainType.UNWALKABLE: [(255, 0, 0), (200, 0, 0)]  # Red (unwalkable)
        }
        
        # Height map (if available)
        self.height_map = None
        
        # Known obstacle positions
        self.known_obstacles = []
        
        self.logger.info("TerrainAnalyzer initialized")
    
    def analyze_terrain(self, state: GameState, position: Position3D, 
                       radius: float = 50.0) -> Dict[Tuple[int, int], TerrainSample]:
        """
        Analyze terrain around a position
        
        Args:
            state: Current game state
            position: Center position for analysis
            radius: Radius to analyze
            
        Returns:
            Dict[Tuple[int, int], TerrainSample]: Grid of terrain samples
        """
        terrain_samples = {}
        
        # Check if we have minimap data
        minimap_data = None
        if hasattr(state, "minimap_data") and state.minimap_data:
            minimap_data = state.minimap_data
        
        # Check if we have screen data
        screenshot = None
        if hasattr(state, "screenshot") and state.screenshot is not None:
            screenshot = state.screenshot
        
        # Grid resolution
        grid_res = 5.0  # 5 units per grid cell
        
        # Create grid bounds
        min_x = int((position.x - radius) // grid_res)
        max_x = int((position.x + radius) // grid_res)
        min_y = int((position.y - radius) // grid_res)
        max_y = int((position.y + radius) // grid_res)
        
        # Create samples for each grid cell
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # Calculate world position
                world_x = x * grid_res
                world_y = y * grid_res
                
                # Skip if too far from center
                dist = math.sqrt((world_x - position.x) ** 2 + (world_y - position.y) ** 2)
                if dist > radius:
                    continue
                
                # Get height at this position (if available)
                height = self._get_height(world_x, world_y, state)
                
                # Calculate grid position
                grid_pos = (x, y)
                
                # Create terrain sample
                sample_pos = Position3D(world_x, world_y, height)
                terrain_samples[grid_pos] = self._analyze_sample(sample_pos, state, screenshot, minimap_data)
        
        return terrain_samples
    
    def _analyze_sample(self, position: Position3D, state: GameState,
                      screenshot: Optional[np.ndarray],
                      minimap_data: Optional[Dict]) -> TerrainSample:
        """
        Analyze a single terrain sample
        
        Args:
            position: Position to analyze
            state: Game state
            screenshot: Game screenshot (if available)
            minimap_data: Minimap data (if available)
            
        Returns:
            TerrainSample: Analyzed terrain sample
        """
        # Default terrain (normal ground)
        terrain_type = TerrainType.NORMAL
        color = (128, 128, 128)  # Default gray
        slope = 0.0
        
        # Check if position is near water (from minimap color)
        if minimap_data and "water_positions" in minimap_data:
            for water_pos, depth in minimap_data["water_positions"]:
                water_x, water_y = water_pos
                dist = math.sqrt((position.x - water_x) ** 2 + (position.y - water_y) ** 2)
                
                if dist < 10.0:  # Within 10 units of known water
                    if depth < 2.0:
                        terrain_type = TerrainType.WATER_SHALLOW
                        color = (180, 180, 255)  # Light blue
                    else:
                        terrain_type = TerrainType.WATER_DEEP
                        color = (0, 0, 200)  # Deep blue
                    break
        
        # Check for obstacles
        for obstacle in self.known_obstacles:
            obs_pos, obs_radius = obstacle
            dist = math.sqrt((position.x - obs_pos.x) ** 2 + (position.y - obs_pos.y) ** 2)
            
            if dist < obs_radius:
                terrain_type = TerrainType.UNWALKABLE
                color = (255, 0, 0)  # Red
                break
        
        # Calculate slope if we have height data
        if self.height_map is not None:
            slope = self._calculate_slope(position.x, position.y)
            
            # Update terrain type based on slope
            if slope > 60.0:
                terrain_type = TerrainType.CLIFF
                color = (80, 60, 40)  # Dark brown
            elif slope > 30.0:
                terrain_type = TerrainType.MOUNTAIN
                color = (160, 120, 80)  # Brown
        
        # Create terrain sample
        sample = TerrainSample(position, terrain_type, color, slope)
        
        return sample
    
    def _get_height(self, x: float, y: float, state: GameState) -> float:
        """
        Get height at a specific position
        
        Args:
            x: X coordinate
            y: Y coordinate
            state: Game state
            
        Returns:
            float: Height at position
        """
        # If we have a height map, use it
        if self.height_map is not None:
            return self._interpolate_height(x, y)
        
        # Otherwise, try to get height from player position
        if hasattr(state, "player_position") and len(state.player_position) >= 3:
            px, py, pz = state.player_position
            
            # Assume flat terrain (same height as player)
            return pz
        
        # Default height (0)
        return 0.0
    
    def _interpolate_height(self, x: float, y: float) -> float:
        """
        Interpolate height from height map
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            float: Interpolated height
        """
        # This is a placeholder for height interpolation
        # In a real implementation, this would use bilinear interpolation
        # from a proper height map
        
        # For now, just return a simple sine wave height
        return 5.0 * math.sin(x / 50.0) * math.cos(y / 50.0)
    
    def _calculate_slope(self, x: float, y: float) -> float:
        """
        Calculate slope angle at a position
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            float: Slope angle in degrees
        """
        # Sample heights in nearby points
        h_center = self._interpolate_height(x, y)
        h_north = self._interpolate_height(x, y + 1)
        h_east = self._interpolate_height(x + 1, y)
        
        # Calculate slope vectors
        dx = h_east - h_center
        dy = h_north - h_center
        
        # Calculate slope magnitude
        slope_mag = math.sqrt(dx * dx + dy * dy)
        
        # Convert to angle in degrees
        slope_angle = math.atan(slope_mag) * 180.0 / math.pi
        
        return slope_angle
    
    def detect_obstacles_from_screen(self, screenshot: np.ndarray, 
                                  player_pos: Position3D) -> List[Tuple[Position3D, float]]:
        """
        Detect obstacles from screen image
        
        Args:
            screenshot: Game screenshot
            player_pos: Player position
            
        Returns:
            List[Tuple[Position3D, float]]: List of (position, radius) tuples
        """
        obstacles = []
        
        # This is a placeholder for obstacle detection
        # In a real implementation, this would use computer vision techniques
        # to detect obstacles from the game's visual output
        
        try:
            # Convert to HSV for color-based detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
            
            # Detect specific object types by color
            # Example: Detect trees (green)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate world position from screen coordinates
                # This would need a proper screen-to-world transformation
                # For now, we'll use a simplified approach
                est_world_x = player_pos.x + (center_x - screenshot.shape[1] // 2) / 10
                est_world_y = player_pos.y + (center_y - screenshot.shape[0] // 2) / 10
                
                # Estimate radius from contour size
                radius = max(w, h) / 20  # Arbitrary scaling
                
                # Create obstacle
                obs_pos = Position3D(est_world_x, est_world_y, player_pos.z)
                obstacles.append((obs_pos, radius))
        
        except Exception as e:
            self.logger.error(f"Error detecting obstacles from screen: {e}")
        
        return obstacles
    
    def detect_terrain_from_minimap(self, minimap: np.ndarray) -> Dict:
        """
        Detect terrain features from minimap
        
        Args:
            minimap: Minimap image
            
        Returns:
            Dict: Detected terrain features
        """
        terrain_features = {
            "water_positions": [],
            "road_positions": [],
            "mountain_positions": []
        }
        
        try:
            # Convert to HSV for color-based detection
            hsv = cv2.cvtColor(minimap, cv2.COLOR_RGB2HSV)
            
            # Detect water (blue)
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Detect roads (gray)
            lower_gray = np.array([0, 0, 100])
            upper_gray = np.array([180, 30, 200])
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            
            # Detect mountains (brown)
            lower_brown = np.array([10, 50, 50])
            upper_brown = np.array([30, 255, 200])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Process water regions
            water_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in water_contours:
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Get center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Estimate depth from color intensity
                    area = cv2.contourArea(contour)
                    depth = min(5.0, area / 1000)  # Arbitrary scaling
                    
                    terrain_features["water_positions"].append(((cx, cy), depth))
            
            # Process road regions
            road_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in road_contours:
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Get center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    terrain_features["road_positions"].append((cx, cy))
            
            # Process mountain regions
            mountain_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in mountain_contours:
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Get center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Estimate height from color intensity
                    area = cv2.contourArea(contour)
                    height = min(50.0, area / 100)  # Arbitrary scaling
                    
                    terrain_features["mountain_positions"].append(((cx, cy), height))
        
        except Exception as e:
            self.logger.error(f"Error detecting terrain from minimap: {e}")
        
        return terrain_features
    
    def identify_jump_paths(self, terrain_samples: Dict[Tuple[int, int], TerrainSample]) -> List[Tuple[Position3D, Position3D]]:
        """
        Identify potential jump paths in terrain
        
        Args:
            terrain_samples: Grid of terrain samples
            
        Returns:
            List[Tuple[Position3D, Position3D]]: List of jump paths (start, end)
        """
        jump_paths = []
        
        # Group samples by z-level
        height_groups = {}
        for grid_pos, sample in terrain_samples.items():
            # Round height to nearest 1.0
            height = round(sample.position.z)
            
            if height not in height_groups:
                height_groups[height] = []
            
            height_groups[height].append(sample)
        
        # Check for potential jumps between different heights
        for h1 in height_groups:
            for h2 in height_groups:
                # Check if heights allow for jumping (down or up moderately)
                height_diff = h2 - h1
                
                if height_diff < -20 or height_diff > 2:
                    # Too high to jump up or too far to jump down
                    continue
                
                # Check each sample at height h1
                for start_sample in height_groups[h1]:
                    # Find samples at height h2 within jump distance
                    for end_sample in height_groups[h2]:
                        # Calculate horizontal distance
                        h_dist = math.sqrt(
                            (start_sample.position.x - end_sample.position.x) ** 2 +
                            (start_sample.position.y - end_sample.position.y) ** 2
                        )
                        
                        # Check if within jump distance
                        if h_dist > 10.0:
                            continue
                        
                        # Check if both positions are walkable
                        if not start_sample.is_walkable() or not end_sample.is_walkable():
                            continue
                        
                        # Add jump path
                        jump_paths.append((start_sample.position, end_sample.position))
        
        return jump_paths
    
    def generate_height_map(self, state: GameState, center: Position3D, 
                          radius: float, resolution: float = 5.0) -> np.ndarray:
        """
        Generate a height map for an area
        
        Args:
            state: Game state
            center: Center position
            radius: Radius to cover
            resolution: Grid resolution
            
        Returns:
            np.ndarray: 2D height map
        """
        # Calculate grid dimensions
        grid_size = int(2 * radius / resolution)
        height_map = np.zeros((grid_size, grid_size), dtype=float)
        
        # Fill in height values
        for i in range(grid_size):
            for j in range(grid_size):
                # Convert grid to world coordinates
                x = center.x - radius + i * resolution
                y = center.y - radius + j * resolution
                
                # Get height at this position
                height = self._get_height(x, y, state)
                
                # Store in height map
                height_map[i, j] = height
        
        # Store for future use
        self.height_map = height_map
        
        return height_map
    
    def find_safe_path(self, start: Position3D, end: Position3D, 
                     terrain_samples: Dict[Tuple[int, int], TerrainSample]) -> List[Position3D]:
        """
        Find a safe path through terrain
        
        Args:
            start: Start position
            end: End position
            terrain_samples: Grid of terrain samples
            
        Returns:
            List[Position3D]: Safe path waypoints
        """
        # This is a simplified version that avoids dangerous terrain
        # In a real implementation, this would use proper pathfinding
        
        # Create grid of walkable vs unwalkable
        min_x = min(s.position.x for s in terrain_samples.values())
        max_x = max(s.position.x for s in terrain_samples.values())
        min_y = min(s.position.y for s in terrain_samples.values())
        max_y = max(s.position.y for s in terrain_samples.values())
        
        grid_res = 5.0  # Assume 5 unit grid resolution
        
        grid_width = int((max_x - min_x) / grid_res) + 1
        grid_height = int((max_y - min_y) / grid_res) + 1
        
        # Create grid
        grid = [[True for _ in range(grid_height)] for _ in range(grid_width)]
        
        # Mark unwalkable cells
        for grid_pos, sample in terrain_samples.items():
            if not sample.is_walkable():
                i = int((sample.position.x - min_x) / grid_res)
                j = int((sample.position.y - min_y) / grid_res)
                
                if 0 <= i < grid_width and 0 <= j < grid_height:
                    grid[i][j] = False
        
        # Convert start and end to grid coordinates
        start_i = int((start.x - min_x) / grid_res)
        start_j = int((start.y - min_y) / grid_res)
        end_i = int((end.x - min_x) / grid_res)
        end_j = int((end.y - min_y) / grid_res)
        
        # Clamp to grid bounds
        start_i = max(0, min(start_i, grid_width - 1))
        start_j = max(0, min(start_j, grid_height - 1))
        end_i = max(0, min(end_i, grid_width - 1))
        end_j = max(0, min(end_j, grid_height - 1))
        
        # A* pathfinding on the grid
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {(start_i, start_j): 0}
        f_score = {(start_i, start_j): self._manhattan_dist((start_i, start_j), (end_i, end_j))}
        
        # Add start to open set
        heapq.heappush(open_set, (f_score[(start_i, start_j)], start_i, start_j))
        
        while open_set:
            # Get node with lowest f_score
            _, i, j = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if i == end_i and j == end_j:
                # Reconstruct path
                path = []
                current = (i, j)
                
                while current in came_from:
                    ci, cj = current
                    world_x = min_x + ci * grid_res
                    world_y = min_y + cj * grid_res
                    
                    # Get height at this position
                    for grid_pos, sample in terrain_samples.items():
                        if int((sample.position.x - min_x) / grid_res) == ci and \
                           int((sample.position.y - min_y) / grid_res) == cj:
                            height = sample.position.z
                            break
                    else:
                        height = 0.0
                    
                    # Add to path
                    path.append(Position3D(world_x, world_y, height))
                    
                    current = came_from[current]
                
                # Add start position
                world_x = min_x + start_i * grid_res
                world_y = min_y + start_j * grid_res
                path.append(Position3D(world_x, world_y, start.z))
                
                # Reverse path
                path.reverse()
                
                # Add end position
                path.append(end)
                
                return path
            
            # Add to closed set
            closed_set.add((i, j))
            
            # Check neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]:
                ni, nj = i + di, j + dj
                
                # Check bounds
                if ni < 0 or ni >= grid_width or nj < 0 or nj >= grid_height:
                    continue
                
                # Check if walkable
                if not grid[ni][nj]:
                    continue
                
                # Skip if in closed set
                if (ni, nj) in closed_set:
                    continue
                
                # Calculate cost
                if di == 0 or dj == 0:
                    cost = 1.0  # Cardinal direction
                else:
                    cost = 1.414  # Diagonal (sqrt(2))
                
                # Calculate tentative g score
                tentative_g = g_score[(i, j)] + cost
                
                # Check if this path is better
                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    # Record this path
                    came_from[(ni, nj)] = (i, j)
                    g_score[(ni, nj)] = tentative_g
                    f_score[(ni, nj)] = tentative_g + self._manhattan_dist((ni, nj), (end_i, end_j))
                    
                    # Add to open set if not already there
                    if not any(n[1] == ni and n[2] == nj for n in open_set):
                        heapq.heappush(open_set, (f_score[(ni, nj)], ni, nj))
        
        # If no path found, return direct line
        return [start, end]
    
    def _manhattan_dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance between grid cells
        
        Args:
            a: First grid cell
            b: Second grid cell
            
        Returns:
            float: Manhattan distance
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])