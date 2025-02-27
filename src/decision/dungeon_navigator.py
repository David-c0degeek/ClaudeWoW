"""
Dungeon Navigator Module

This module provides specialized navigation for dungeons and raids:
- Dungeon map processing and pathing
- Boss and points of interest navigation
- Multi-level navigation
- Specialized dungeon movement techniques
"""

import logging
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set, Union

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.advanced_navigation import Position3D, TerrainNode, TerrainType, TerrainMap

class DungeonArea:
    """Represents an area within a dungeon"""
    
    def __init__(self, name: str, level: int = 0):
        """
        Initialize a dungeon area
        
        Args:
            name: Name of the area
            level: Vertical level/floor in the dungeon
        """
        self.name = name
        self.level = level
        self.points_of_interest = {}  # name -> Position3D
        self.connected_areas = {}  # name -> connection_point
        self.terrain_map = None  # TerrainMap for this area
    
    def add_point_of_interest(self, name: str, position: Position3D, poi_type: str = "generic") -> None:
        """
        Add a point of interest
        
        Args:
            name: POI name
            position: POI position
            poi_type: Type of POI (boss, chest, etc.)
        """
        self.points_of_interest[name] = {
            "position": position,
            "type": poi_type
        }
    
    def add_connection(self, target_area: str, position: Position3D, 
                     connection_type: str = "passage") -> None:
        """
        Add a connection to another area
        
        Args:
            target_area: Name of target area
            position: Position of the connection point
            connection_type: Type of connection (passage, portal, etc.)
        """
        self.connected_areas[target_area] = {
            "position": position,
            "type": connection_type
        }
    
    def get_point_of_interest(self, name: str) -> Optional[Dict]:
        """
        Get a point of interest by name
        
        Args:
            name: POI name
            
        Returns:
            Optional[Dict]: POI info or None
        """
        return self.points_of_interest.get(name)
    
    def get_all_points_of_interest(self) -> List[Dict]:
        """
        Get all points of interest
        
        Returns:
            List[Dict]: All POIs with names
        """
        return [
            {"name": name, **poi} 
            for name, poi in self.points_of_interest.items()
        ]
    
    def get_connection(self, target_area: str) -> Optional[Dict]:
        """
        Get connection to another area
        
        Args:
            target_area: Target area name
            
        Returns:
            Optional[Dict]: Connection info or None
        """
        return self.connected_areas.get(target_area)

class DungeonMap:
    """Represents a complete dungeon map with multiple areas"""
    
    def __init__(self, name: str, dungeon_type: str = "dungeon"):
        """
        Initialize a dungeon map
        
        Args:
            name: Dungeon name
            dungeon_type: Type of dungeon (dungeon, raid)
        """
        self.name = name
        self.dungeon_type = dungeon_type
        self.areas = {}  # name -> DungeonArea
        self.entrance_area = None  # Name of entrance area
        self.entrance_position = None  # Position of entrance
        self.boss_areas = {}  # boss_name -> area_name
    
    def add_area(self, area: DungeonArea) -> None:
        """
        Add an area to the dungeon map
        
        Args:
            area: DungeonArea to add
        """
        self.areas[area.name] = area
    
    def set_entrance(self, area_name: str, position: Position3D) -> None:
        """
        Set the dungeon entrance
        
        Args:
            area_name: Name of entrance area
            position: Position of entrance
        """
        self.entrance_area = area_name
        self.entrance_position = position
    
    def add_boss(self, boss_name: str, area_name: str, position: Position3D) -> None:
        """
        Add a boss to the dungeon
        
        Args:
            boss_name: Name of the boss
            area_name: Name of the area containing the boss
            position: Position of the boss
        """
        self.boss_areas[boss_name] = area_name
        
        # Add as point of interest in the area
        if area_name in self.areas:
            self.areas[area_name].add_point_of_interest(
                boss_name, position, "boss")
    
    def find_path_between_areas(self, start_area: str, end_area: str) -> List[Dict]:
        """
        Find a path between two areas
        
        Args:
            start_area: Starting area name
            end_area: Ending area name
            
        Returns:
            List[Dict]: Path as a list of area transitions
        """
        # BFS to find path
        visited = {start_area}
        queue = [[start_area]]
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            if current == end_area:
                # Convert path to area transitions
                transitions = []
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    
                    # Get connection details
                    if source in self.areas:
                        connection = self.areas[source].get_connection(target)
                        if connection:
                            transitions.append({
                                "source_area": source,
                                "target_area": target,
                                "position": connection["position"].to_tuple(),
                                "type": connection["type"]
                            })
                
                return transitions
            
            # Check connected areas
            if current in self.areas:
                for target in self.areas[current].connected_areas:
                    if target not in visited:
                        visited.add(target)
                        queue.append(path + [target])
        
        return []  # No path found

class DungeonNavigator:
    """
    Specialized navigator for dungeons and raids
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the DungeonNavigator
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.dungeon_navigator")
        self.config = config
        self.knowledge = knowledge
        
        # Dungeon maps
        self.dungeon_maps = {}  # name -> DungeonMap
        
        # Current dungeon state
        self.current_dungeon = None
        self.current_area = None
        
        # Initialize dungeon data
        self._init_dungeon_data()
        
        self.logger.info("DungeonNavigator initialized")
    
    def _init_dungeon_data(self) -> None:
        """Initialize dungeon data from knowledge base"""
        # Check if we have instance data
        if not hasattr(self.knowledge, 'instances'):
            self.logger.warning("No instance data found in knowledge base")
            return
        
        # Process each instance
        for instance_id, instance_data in self.knowledge.instances.items():
            instance_name = instance_data.get("name", instance_id)
            instance_type = instance_data.get("type", "dungeon")
            
            # Create dungeon map
            dungeon_map = DungeonMap(instance_name, instance_type)
            
            # Process entrance data
            entrance = instance_data.get("entrance", {})
            entrance_area = entrance.get("area", "entrance")
            entrance_pos = entrance.get("position", [0, 0, 0])
            
            # Create entrance area
            entrance_area_obj = DungeonArea(entrance_area, 0)
            dungeon_map.add_area(entrance_area_obj)
            
            # Set entrance
            if len(entrance_pos) >= 3:
                entrance_position = Position3D(entrance_pos[0], entrance_pos[1], entrance_pos[2])
            else:
                entrance_position = Position3D(entrance_pos[0], entrance_pos[1], 0)
            
            dungeon_map.set_entrance(entrance_area, entrance_position)
            
            # Process bosses
            bosses = instance_data.get("bosses", [])
            for boss_data in bosses:
                boss_name = boss_data.get("name", "")
                boss_area = boss_data.get("area", entrance_area)
                boss_pos = boss_data.get("position", [0, 0, 0])
                
                if not boss_name:
                    continue
                
                # Ensure area exists
                if boss_area not in dungeon_map.areas:
                    boss_area_obj = DungeonArea(boss_area, 0)
                    dungeon_map.add_area(boss_area_obj)
                
                # Add boss
                if len(boss_pos) >= 3:
                    boss_position = Position3D(boss_pos[0], boss_pos[1], boss_pos[2])
                else:
                    boss_position = Position3D(boss_pos[0], boss_pos[1], 0)
                
                dungeon_map.add_boss(boss_name, boss_area, boss_position)
            
            # Process areas
            areas = instance_data.get("areas", [])
            for area_data in areas:
                area_name = area_data.get("name", "")
                area_level = area_data.get("level", 0)
                
                if not area_name or area_name in dungeon_map.areas:
                    continue
                
                # Create area
                area = DungeonArea(area_name, area_level)
                
                # Add points of interest
                for poi in area_data.get("points_of_interest", []):
                    poi_name = poi.get("name", "")
                    poi_type = poi.get("type", "generic")
                    poi_pos = poi.get("position", [0, 0, 0])
                    
                    if not poi_name:
                        continue
                    
                    if len(poi_pos) >= 3:
                        poi_position = Position3D(poi_pos[0], poi_pos[1], poi_pos[2])
                    else:
                        poi_position = Position3D(poi_pos[0], poi_pos[1], 0)
                    
                    area.add_point_of_interest(poi_name, poi_position, poi_type)
                
                # Add connections
                for conn in area_data.get("connections", []):
                    target = conn.get("target", "")
                    conn_type = conn.get("type", "passage")
                    conn_pos = conn.get("position", [0, 0, 0])
                    
                    if not target:
                        continue
                    
                    if len(conn_pos) >= 3:
                        conn_position = Position3D(conn_pos[0], conn_pos[1], conn_pos[2])
                    else:
                        conn_position = Position3D(conn_pos[0], conn_pos[1], 0)
                    
                    area.add_connection(target, conn_position, conn_type)
                
                # Add area to dungeon map
                dungeon_map.add_area(area)
            
            # Add connections from instance data
            connections = instance_data.get("connections", [])
            for conn in connections:
                source = conn.get("source", "")
                target = conn.get("target", "")
                conn_type = conn.get("type", "passage")
                conn_pos = conn.get("position", [0, 0, 0])
                
                if not source or not target:
                    continue
                
                # Ensure both areas exist
                if source not in dungeon_map.areas:
                    source_area = DungeonArea(source, 0)
                    dungeon_map.add_area(source_area)
                
                if target not in dungeon_map.areas:
                    target_area = DungeonArea(target, 0)
                    dungeon_map.add_area(target_area)
                
                # Add connection
                if len(conn_pos) >= 3:
                    conn_position = Position3D(conn_pos[0], conn_pos[1], conn_pos[2])
                else:
                    conn_position = Position3D(conn_pos[0], conn_pos[1], 0)
                
                dungeon_map.areas[source].add_connection(target, conn_position, conn_type)
            
            # Add dungeon map
            self.dungeon_maps[instance_name.lower()] = dungeon_map
            self.logger.info(f"Initialized dungeon map for {instance_name}")
    
    def get_dungeon_map(self, dungeon_name: str) -> Optional[DungeonMap]:
        """
        Get a dungeon map by name
        
        Args:
            dungeon_name: Name of the dungeon
            
        Returns:
            Optional[DungeonMap]: Dungeon map or None
        """
        return self.dungeon_maps.get(dungeon_name.lower())
    
    def is_in_dungeon(self, state: GameState) -> bool:
        """
        Check if player is in a dungeon
        
        Args:
            state: Current game state
            
        Returns:
            bool: True if in dungeon
        """
        # Check if zone is an instance
        if hasattr(state, "current_zone"):
            zone = state.current_zone.lower()
            
            # Check each dungeon name
            for dungeon_name in self.dungeon_maps:
                if dungeon_name in zone:
                    return True
        
        # Check for instance-specific UI elements
        if hasattr(state, "ui_state") and state.ui_state:
            # Check for instance map
            if state.ui_state.get("instance_map_visible", False):
                return True
        
        return False
    
    def detect_current_dungeon(self, state: GameState) -> Optional[str]:
        """
        Detect the current dungeon from the game state
        
        Args:
            state: Current game state
            
        Returns:
            Optional[str]: Dungeon name or None
        """
        # Check zone name
        if hasattr(state, "current_zone"):
            zone = state.current_zone.lower()
            
            for dungeon_name in self.dungeon_maps:
                if dungeon_name in zone:
                    self.current_dungeon = dungeon_name
                    self.logger.info(f"Detected dungeon: {dungeon_name}")
                    return dungeon_name
        
        # Check for on-screen text that might indicate the dungeon
        if hasattr(state, "text_elements") and state.text_elements:
            for text in state.text_elements:
                if "text" in text:
                    text_content = text["text"].lower()
                    
                    for dungeon_name in self.dungeon_maps:
                        if dungeon_name in text_content:
                            self.current_dungeon = dungeon_name
                            self.logger.info(f"Detected dungeon from text: {dungeon_name}")
                            return dungeon_name
        
        # No dungeon detected
        self.current_dungeon = None
        return None
    
    def detect_current_area(self, state: GameState) -> Optional[str]:
        """
        Detect the current area within the dungeon
        
        Args:
            state: Current game state
            
        Returns:
            Optional[str]: Area name or None
        """
        # Ensure we have a current dungeon
        if not self.current_dungeon:
            self.detect_current_dungeon(state)
            
            if not self.current_dungeon:
                return None
        
        dungeon_map = self.dungeon_maps.get(self.current_dungeon)
        if not dungeon_map:
            return None
        
        # Get player position
        player_pos = None
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
        
        if not player_pos:
            return None
        
        # Check subzone name if available
        if hasattr(state, "current_subzone") and state.current_subzone:
            subzone = state.current_subzone.lower()
            
            # Check if subzone matches any area name
            for area_name, area in dungeon_map.areas.items():
                if area_name.lower() in subzone:
                    self.current_area = area_name
                    return area_name
        
        # If no subzone match, try to determine area by proximity to known points
        best_area = None
        min_distance = float('inf')
        
        for area_name, area in dungeon_map.areas.items():
            # Check points of interest
            for poi_name, poi_data in area.points_of_interest.items():
                poi_pos = poi_data["position"]
                distance = player_pos.distance_to(poi_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    best_area = area_name
            
            # Check connections
            for target, conn_data in area.connected_areas.items():
                conn_pos = conn_data["position"]
                distance = player_pos.distance_to(conn_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    best_area = area_name
        
        if best_area and min_distance < 100:  # Within 100 units of a known point
            self.current_area = best_area
            return best_area
        
        # Default to entrance area if we can't determine current area
        if dungeon_map.entrance_area:
            self.current_area = dungeon_map.entrance_area
            return dungeon_map.entrance_area
        
        return None
    
    def navigate_to_boss(self, state: GameState, boss_name: str) -> List[Dict]:
        """
        Navigate to a boss in the current dungeon
        
        Args:
            state: Current game state
            boss_name: Name of the boss
            
        Returns:
            List[Dict]: Navigation actions
        """
        # Ensure we have current dungeon and area
        if not self.current_dungeon:
            self.detect_current_dungeon(state)
            
            if not self.current_dungeon:
                self.logger.warning("Cannot navigate to boss: unknown dungeon")
                return []
        
        if not self.current_area:
            self.detect_current_area(state)
            
            if not self.current_area:
                self.logger.warning("Cannot navigate to boss: unknown area")
                return []
        
        dungeon_map = self.dungeon_maps.get(self.current_dungeon)
        if not dungeon_map:
            self.logger.warning(f"No map for dungeon: {self.current_dungeon}")
            return []
        
        # Find boss area
        boss_area = None
        boss_position = None
        
        # Normalize boss name for comparison
        boss_name_lower = boss_name.lower()
        
        for b_name, area_name in dungeon_map.boss_areas.items():
            if boss_name_lower in b_name.lower():
                boss_area = area_name
                
                # Get boss position
                area = dungeon_map.areas.get(area_name)
                if area:
                    boss_poi = area.get_point_of_interest(b_name)
                    if boss_poi:
                        boss_position = boss_poi["position"]
                
                break
        
        if not boss_area or not boss_position:
            self.logger.warning(f"Boss not found: {boss_name}")
            return []
        
        # Check if already in boss area
        if self.current_area == boss_area:
            # Navigate directly to boss
            return [{
                "type": "move",
                "position": boss_position.to_tuple(),
                "description": f"Move to boss: {boss_name}"
            }]
        
        # Find path between areas
        area_path = dungeon_map.find_path_between_areas(self.current_area, boss_area)
        
        if not area_path:
            self.logger.warning(f"No path found to boss area: {boss_area}")
            return []
        
        # Create navigation actions
        actions = []
        
        # Add actions to navigate through each area
        for transition in area_path:
            actions.append({
                "type": "move",
                "position": transition["position"],
                "description": f"Move to {transition['type']} to {transition['target_area']}"
            })
            
            if transition["type"] == "portal":
                actions.append({
                    "type": "use_portal",
                    "description": f"Use portal to {transition['target_area']}"
                })
            elif transition["type"] == "door":
                actions.append({
                    "type": "open_door",
                    "description": f"Open door to {transition['target_area']}"
                })
        
        # Finally, navigate to boss
        actions.append({
            "type": "move",
            "position": boss_position.to_tuple(),
            "description": f"Move to boss: {boss_name}"
        })
        
        return actions
    
    def get_dungeon_overview(self, dungeon_name: str) -> Optional[Dict]:
        """
        Get an overview of a dungeon
        
        Args:
            dungeon_name: Name of the dungeon
            
        Returns:
            Optional[Dict]: Dungeon overview or None
        """
        # Get dungeon details from knowledge
        instance_info = None
        if hasattr(self.knowledge, 'get_instance_info'):
            instance_info = self.knowledge.get_instance_info(dungeon_name)
        
        if not instance_info:
            self.logger.warning(f"No instance info for: {dungeon_name}")
            return None
        
        # Get dungeon map
        dungeon_map = self.dungeon_maps.get(dungeon_name.lower())
        if not dungeon_map:
            self.logger.warning(f"No map for dungeon: {dungeon_name}")
            return None
        
        # Create overview
        overview = {
            "name": dungeon_map.name,
            "type": dungeon_map.dungeon_type,
            "level_range": instance_info.get("level_range", [0, 0]),
            "areas": [],
            "bosses": [],
            "tips": instance_info.get("tips", [])
        }
        
        # Add areas
        for area_name, area in dungeon_map.areas.items():
            area_overview = {
                "name": area_name,
                "level": area.level,
                "points_of_interest": area.get_all_points_of_interest()
            }
            overview["areas"].append(area_overview)
        
        # Add bosses
        for boss_name, area_name in dungeon_map.boss_areas.items():
            # Find boss data in instance info
            boss_data = None
            for b in instance_info.get("bosses", []):
                if b.get("name", "") == boss_name:
                    boss_data = b
                    break
            
            if not boss_data:
                continue
            
            # Add boss overview
            boss_overview = {
                "name": boss_name,
                "area": area_name,
                "strategy": boss_data.get("strategy", "")
            }
            overview["bosses"].append(boss_overview)
        
        return overview
    
    def get_next_boss(self, state: GameState) -> Optional[Dict]:
        """
        Get the next boss in the dungeon
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Dict]: Next boss info or None
        """
        # Ensure we have current dungeon and area
        if not self.current_dungeon:
            self.detect_current_dungeon(state)
            
            if not self.current_dungeon:
                return None
        
        if not self.current_area:
            self.detect_current_area(state)
            
            if not self.current_area:
                return None
        
        dungeon_map = self.dungeon_maps.get(self.current_dungeon)
        if not dungeon_map:
            return None
        
        # Get dungeon details from knowledge
        instance_info = None
        if hasattr(self.knowledge, 'get_instance_info'):
            instance_info = self.knowledge.get_instance_info(self.current_dungeon)
        
        if not instance_info:
            return None
        
        # Get the boss order from instance info
        boss_order = []
        for boss in instance_info.get("bosses", []):
            boss_name = boss.get("name", "")
            if boss_name:
                boss_order.append(boss_name)
        
        # Track killed bosses (this would need to be updated based on game state)
        killed_bosses = set()
        if hasattr(state, "killed_bosses"):
            killed_bosses = set(state.killed_bosses)
        
        # Find the next boss in order
        for boss_name in boss_order:
            if boss_name not in killed_bosses and boss_name in dungeon_map.boss_areas:
                # Get boss area
                area_name = dungeon_map.boss_areas[boss_name]
                area = dungeon_map.areas.get(area_name)
                
                if not area:
                    continue
                
                # Get boss position
                boss_poi = area.get_point_of_interest(boss_name)
                if not boss_poi:
                    continue
                
                # Get boss strategy
                strategy = ""
                for boss_data in instance_info.get("bosses", []):
                    if boss_data.get("name", "") == boss_name:
                        strategy = boss_data.get("strategy", "")
                        break
                
                return {
                    "name": boss_name,
                    "area": area_name,
                    "position": boss_poi["position"].to_tuple(),
                    "strategy": strategy
                }
        
        return None  # No next boss found
    
    def find_nearest_point_of_interest(self, state: GameState, poi_type: str = None) -> Optional[Dict]:
        """
        Find the nearest point of interest in the dungeon
        
        Args:
            state: Current game state
            poi_type: Type of POI to look for (optional)
            
        Returns:
            Optional[Dict]: Nearest POI or None
        """
        # Ensure we have current dungeon and area
        if not self.current_dungeon:
            self.detect_current_dungeon(state)
            
            if not self.current_dungeon:
                return None
        
        if not self.current_area:
            self.detect_current_area(state)
            
            if not self.current_area:
                return None
        
        dungeon_map = self.dungeon_maps.get(self.current_dungeon)
        if not dungeon_map:
            return None
        
        # Get player position
        player_pos = None
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
        
        if not player_pos:
            return None
        
        # Find the nearest POI
        nearest_poi = None
        min_distance = float('inf')
        
        # Check POIs in current area first
        current_area = dungeon_map.areas.get(self.current_area)
        if current_area:
            for poi_name, poi_data in current_area.points_of_interest.items():
                # Skip if filtering by type and doesn't match
                if poi_type and poi_data["type"] != poi_type:
                    continue
                
                distance = player_pos.distance_to(poi_data["position"])
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_poi = {
                        "name": poi_name,
                        "type": poi_data["type"],
                        "position": poi_data["position"].to_tuple(),
                        "area": self.current_area,
                        "distance": distance
                    }
        
        # If no POI found in current area, check other areas
        if nearest_poi is None:
            for area_name, area in dungeon_map.areas.items():
                if area_name == self.current_area:
                    continue
                
                for poi_name, poi_data in area.points_of_interest.items():
                    # Skip if filtering by type and doesn't match
                    if poi_type and poi_data["type"] != poi_type:
                        continue
                    
                    # This is a rough estimate across areas
                    distance = player_pos.distance_to(poi_data["position"]) + 100  # Add penalty for area change
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_poi = {
                            "name": poi_name,
                            "type": poi_data["type"],
                            "position": poi_data["position"].to_tuple(),
                            "area": area_name,
                            "distance": distance
                        }
        
        return nearest_poi