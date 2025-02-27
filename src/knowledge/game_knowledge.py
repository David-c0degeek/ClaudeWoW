"""
Game Knowledge Module

This module serves as a knowledge base for game information.
"""

import logging
import json
import os
import math
import time
from typing import Dict, List, Tuple, Any, Optional, Set

class GameKnowledge:
    """
    Knowledge base for World of Warcraft game information
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GameKnowledge
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.knowledge.game_knowledge")
        self.config = config
        
        # Initialize knowledge databases
        self.zones = {}
        self.npcs = {}
        self.quests = {}
        self.abilities = {}
        self.items = {}
        self.paths = {}
        
        # Initialize knowledge tracking
        self.known_zones = set()
        self.known_npcs = set()
        self.known_quests = set()
        
        # Load knowledge from files
        self._load_knowledge()
        
        # Runtime knowledge (learned during execution)
        self.runtime_knowledge = {
            "zones": {},
            "npcs": {},
            "quests": {},
            "paths": {}
        }
        
        self.logger.info("GameKnowledge initialized")
    
    def _load_knowledge(self) -> None:
        """
        Load knowledge from data files
        """
        # Get data directory
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "game_knowledge"
        )
        
        # Create directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            self.logger.info(f"Created game knowledge directory: {data_dir}")
        
        # Load zones data
        self._load_json_file(os.path.join(data_dir, "zones.json"), "zones")
        
        # Load NPCs data
        self._load_json_file(os.path.join(data_dir, "npcs.json"), "npcs")
        
        # Load quests data
        self._load_json_file(os.path.join(data_dir, "quests.json"), "quests")
        
        # Load abilities data
        self._load_json_file(os.path.join(data_dir, "abilities.json"), "abilities")
        
        # Load items data
        self._load_json_file(os.path.join(data_dir, "items.json"), "items")
        
        # Load paths data
        self._load_json_file(os.path.join(data_dir, "paths.json"), "paths")
        
        # Load instances data
        self._load_json_file(os.path.join(data_dir, "instances.json"), "instances")
    
        # Load emotes data
        self._load_json_file(os.path.join(data_dir, "emotes.json"), "emotes")
    
        # Load social responses data
        self._load_json_file(os.path.join(data_dir, "social_responses.json"), "social_responses")

    
    def _load_json_file(self, file_path: str, data_type: str) -> None:
        """
        Load data from a JSON file
        
        Args:
            file_path: Path to the JSON file
            data_type: Type of data being loaded
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Store data in the appropriate attribute
                if data_type == "zones":
                    self.zones = data
                elif data_type == "npcs":
                    self.npcs = data
                elif data_type == "quests":
                    self.quests = data
                elif data_type == "abilities":
                    self.abilities = data
                elif data_type == "items":
                    self.items = data
                elif data_type == "paths":
                    self.paths = data
                
                self.logger.info(f"Loaded {len(data)} {data_type} from {file_path}")
            else:
                # Create empty data file
                if data_type == "zones":
                    self._create_default_zones_data(file_path)
                elif data_type == "npcs":
                    self._create_default_npcs_data(file_path)
                elif data_type == "quests":
                    self._create_default_quests_data(file_path)
                elif data_type == "abilities":
                    self._create_default_abilities_data(file_path)
                elif data_type == "items":
                    self._create_default_items_data(file_path)
                elif data_type == "paths":
                    self._create_default_paths_data(file_path)
        
        except Exception as e:
            self.logger.error(f"Error loading {data_type} data from {file_path}: {e}")
    
    def _create_default_zones_data(self, file_path: str) -> None:
        """
        Create default zones data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "elwynn_forest": {
                "name": "Elwynn Forest",
                "level_range": [1, 10],
                "faction": "alliance",
                "neighbors": ["westfall", "duskwood", "redridge_mountains"],
                "main_city": "stormwind",
                "flight_paths": ["stormwind"],
                "quest_hubs": ["northshire", "goldshire"],
                "points_of_interest": [
                    {"name": "Northshire Abbey", "position": [0, 0]},
                    {"name": "Goldshire", "position": [100, 100]},
                    {"name": "Stormwind", "position": [200, 200]}
                ]
            },
            "durotar": {
                "name": "Durotar",
                "level_range": [1, 10],
                "faction": "horde",
                "neighbors": ["the_barrens", "orgrimmar"],
                "main_city": "orgrimmar",
                "flight_paths": ["orgrimmar"],
                "quest_hubs": ["valley_of_trials", "razor_hill"],
                "points_of_interest": [
                    {"name": "Valley of Trials", "position": [0, 0]},
                    {"name": "Razor Hill", "position": [100, 100]},
                    {"name": "Orgrimmar", "position": [200, 200]}
                ]
            }
        }
        
        self.zones = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default zones data at {file_path}")
    
    def _create_default_npcs_data(self, file_path: str) -> None:
        """
        Create default NPCs data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "marshal_mcbride": {
                "id": "marshal_mcbride",
                "name": "Marshal McBride",
                "type": "quest_giver",
                "faction": "alliance",
                "location": "northshire",
                "position": [0, 0],
                "gives_quests": ["a_threat_within"],
                "accepts_quests": []
            },
            "deputy_willem": {
                "id": "deputy_willem",
                "name": "Deputy Willem",
                "type": "quest_giver",
                "faction": "alliance",
                "location": "northshire",
                "position": [10, 10],
                "gives_quests": ["wolves_across_the_border"],
                "accepts_quests": ["wolves_across_the_border"]
            },
            "inn_keeper_farley": {
                "id": "inn_keeper_farley",
                "name": "Innkeeper Farley",
                "type": "innkeeper",
                "faction": "alliance",
                "location": "goldshire",
                "position": [100, 100],
                "gives_quests": [],
                "accepts_quests": []
            },
            "smith_argus": {
                "id": "smith_argus",
                "name": "Smith Argus",
                "type": "vendor",
                "faction": "alliance",
                "location": "goldshire",
                "position": [110, 100],
                "gives_quests": [],
                "accepts_quests": []
            }
        }
        
        self.npcs = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default NPCs data at {file_path}")
    
    def _create_default_quests_data(self, file_path: str) -> None:
        """
        Create default quests data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "a_threat_within": {
                "id": "a_threat_within",
                "title": "A Threat Within",
                "level": 1,
                "faction": "alliance",
                "zone": "elwynn_forest",
                "location": "northshire",
                "quest_giver": "marshal_mcbride",
                "turn_in": "marshal_mcbride",
                "pre_requisites": [],
                "follow_up": ["beating_them_back"],
                "description": "Speak with Marshal McBride.",
                "objectives": [
                    {
                        "type": "interact",
                        "target": "marshal_mcbride",
                        "count": 1,
                        "description": "Speak with Marshal McBride"
                    }
                ],
                "rewards": {
                    "xp": 40,
                    "items": []
                }
            },
            "wolves_across_the_border": {
                "id": "wolves_across_the_border",
                "title": "Wolves Across the Border",
                "level": 1,
                "faction": "alliance",
                "zone": "elwynn_forest",
                "location": "northshire",
                "quest_giver": "deputy_willem",
                "turn_in": "deputy_willem",
                "pre_requisites": [],
                "follow_up": [],
                "description": "Kill Timber Wolves and bring their meat to Deputy Willem.",
                "objectives": [
                    {
                        "type": "kill",
                        "target": "timber_wolf",
                        "count": 8,
                        "description": "Kill 8 Timber Wolves"
                    },
                    {
                        "type": "collect",
                        "item": "wolf_meat",
                        "count": 8,
                        "description": "Collect 8 Wolf Meat",
                        "source": "mob",
                        "mob": "timber_wolf"
                    }
                ],
                "rewards": {
                    "xp": 100,
                    "items": [
                        {"id": "worn_short_sword", "chance": 1.0}
                    ]
                }
            }
        }
        
        self.quests = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default quests data at {file_path}")
    
    def _create_default_abilities_data(self, file_path: str) -> None:
        """
        Create default abilities data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "warrior": {
                "battle_shout": {
                    "name": "Battle Shout",
                    "rank": 1,
                    "level": 1,
                    "type": "buff",
                    "resource": "rage",
                    "cost": 10,
                    "cooldown": 0,
                    "effects": [
                        {"type": "buff", "target": "party", "stat": "attack_power", "value": 25, "duration": 120}
                    ]
                },
                "heroic_strike": {
                    "name": "Heroic Strike",
                    "rank": 1,
                    "level": 1,
                    "type": "ability",
                    "resource": "rage",
                    "cost": 15,
                    "cooldown": 0,
                    "effects": [
                        {"type": "damage", "target": "enemy", "value": 11, "bonus_damage": 1.0}
                    ]
                },
                "charge": {
                    "name": "Charge",
                    "rank": 1,
                    "level": 4,
                    "type": "ability",
                    "resource": "none",
                    "cost": 0,
                    "cooldown": 15,
                    "effects": [
                        {"type": "movement", "distance": 25},
                        {"type": "stun", "duration": 1},
                        {"type": "resource", "resource": "rage", "value": 15}
                    ]
                }
            },
            "paladin": {
                "seal_of_righteousness": {
                    "name": "Seal of Righteousness",
                    "rank": 1,
                    "level": 1,
                    "type": "buff",
                    "resource": "mana",
                    "cost": 50,
                    "cooldown": 0,
                    "effects": [
                        {"type": "buff", "target": "self", "stat": "holy_damage", "value": 10, "duration": 30}
                    ]
                },
                "judgement": {
                    "name": "Judgement",
                    "rank": 1,
                    "level": 4,
                    "type": "ability",
                    "resource": "mana",
                    "cost": 40,
                    "cooldown": 10,
                    "effects": [
                        {"type": "damage", "target": "enemy", "value": 20, "damage_type": "holy"}
                    ]
                }
            }
        }
        
        self.abilities = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default abilities data at {file_path}")
    
    def _create_default_items_data(self, file_path: str) -> None:
        """
        Create default items data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "wolf_meat": {
                "id": "wolf_meat",
                "name": "Wolf Meat",
                "type": "reagent",
                "quality": "common",
                "level": 1,
                "source": ["timber_wolf"],
                "drop_chance": 0.75,
                "value": 5
            },
            "worn_short_sword": {
                "id": "worn_short_sword",
                "name": "Worn Short Sword",
                "type": "weapon",
                "subtype": "one_handed_sword",
                "quality": "common",
                "level": 1,
                "stats": {
                    "damage_min": 2,
                    "damage_max": 4,
                    "speed": 1.6
                },
                "value": 38
            }
        }
        
        self.items = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default items data at {file_path}")
    
    def _create_default_paths_data(self, file_path: str) -> None:
        """
        Create default paths data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "northshire_to_goldshire": {
                "start": "northshire",
                "end": "goldshire",
                "waypoints": [
                    [0, 0],
                    [50, 50],
                    [100, 100]
                ]
            },
            "goldshire_to_stormwind": {
                "start": "goldshire",
                "end": "stormwind",
                "waypoints": [
                    [100, 100],
                    [150, 150],
                    [200, 200]
                ]
            }
        }
        
        self.paths = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default paths data at {file_path}")
    
    def update(self, state: Any) -> None:
        """
        Update knowledge base with new information from game state
        
        Args:
            state: Current game state
        """
        # Update known zones
        if hasattr(state, "current_zone") and state.current_zone:
            self._update_zone_knowledge(state.current_zone, state)
        
        # Update known NPCs
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                if entity.get("type") == "npc":
                    self._update_npc_knowledge(entity, state)
        
        # Update known quests
        if hasattr(state, "active_quests") and state.active_quests:
            for quest in state.active_quests:
                self._update_quest_knowledge(quest, state)
    
    def _update_zone_knowledge(self, zone_name: str, state: Any) -> None:
        """
        Update knowledge about a zone
        
        Args:
            zone_name: Name of the zone
            state: Current game state
        """
        # Check if we already know this zone
        if zone_name in self.known_zones:
            return
        
        # Add to known zones
        self.known_zones.add(zone_name)
        
        # Check if zone is in our database
        if zone_name not in self.zones and zone_name not in self.runtime_knowledge["zones"]:
            # Create new zone entry in runtime knowledge
            self.runtime_knowledge["zones"][zone_name] = {
                "name": zone_name,
                "level_range": [1, 60],  # Default level range
                "faction": "neutral",    # Default faction
                "neighbors": [],
                "main_city": "",
                "flight_paths": [],
                "quest_hubs": [],
                "points_of_interest": []
            }
            
            self.logger.info(f"Added new zone to knowledge base: {zone_name}")
    
    def _update_npc_knowledge(self, npc: Dict, state: Any) -> None:
        """
        Update knowledge about an NPC
        
        Args:
            npc: NPC entity data
            state: Current game state
        """
        npc_id = npc.get("id", "")
        
        if not npc_id:
            return
        
        # Check if NPC is already known
        if npc_id in self.known_npcs:
            return
        
        # Add to known NPCs
        self.known_npcs.add(npc_id)
        
        # Check if NPC is in our database
        if npc_id not in self.npcs and npc_id not in self.runtime_knowledge["npcs"]:
            # Create new NPC entry in runtime knowledge
            current_zone = state.current_zone if hasattr(state, "current_zone") else ""
            
            self.runtime_knowledge["npcs"][npc_id] = {
                "id": npc_id,
                "name": npc.get("name", npc_id),
                "type": npc.get("subtype", "npc"),
                "faction": npc.get("faction", "neutral"),
                "location": current_zone,
                "position": npc.get("position", [0, 0]),
                "gives_quests": [],
                "accepts_quests": []
            }
            
            self.logger.info(f"Added new NPC to knowledge base: {npc_id}")
    
    def _update_quest_knowledge(self, quest: Dict, state: Any) -> None:
        """
        Update knowledge about a quest
        
        Args:
            quest: Quest data
            state: Current game state
        """
        quest_title = quest.get("title", "")
        
        if not quest_title:
            return
        
        # Check if quest is already known
        if quest_title in self.known_quests:
            return
        
        # Add to known quests
        self.known_quests.add(quest_title)
        
        # Check if quest is in our database
        if quest_title not in self.quests and quest_title not in self.runtime_knowledge["quests"]:
            # Create new quest entry in runtime knowledge
            current_zone = state.current_zone if hasattr(state, "current_zone") else ""
            
            self.runtime_knowledge["quests"][quest_title] = {
                "id": quest_title.lower().replace(" ", "_"),
                "title": quest_title,
                "level": state.player_level if hasattr(state, "player_level") else 1,
                "faction": "neutral",
                "zone": current_zone,
                "location": "",
                "quest_giver": "",
                "turn_in": "",
                "pre_requisites": [],
                "follow_up": [],
                "description": "",
                "objectives": [],
                "rewards": {
                    "xp": 0,
                    "items": []
                }
            }
            
            # Try to extract objectives
            if "objectives" in quest:
                objectives = []
                for obj in quest["objectives"]:
                    obj_name = obj.get("name", "")
                    current = obj.get("current", 0)
                    total = obj.get("total", 1)
                    
                    objectives.append({
                        "type": "unknown",
                        "description": obj_name,
                        "count": total
                    })
                
                self.runtime_knowledge["quests"][quest_title]["objectives"] = objectives
            
            self.logger.info(f"Added new quest to knowledge base: {quest_title}")
    
    def get_combat_rotation(self, player_class: str, player_level: int) -> List[Dict]:
        """
        Get the optimal combat rotation for a class and level
        
        Args:
            player_class: Player's class
            player_level: Player's level
        
        Returns:
            List[Dict]: List of abilities to use in order
        """
        # Convert to lowercase for consistency
        player_class = player_class.lower()
        
        # Check if we have ability data for this class
        if player_class not in self.abilities:
            return []
        
        class_abilities = self.abilities[player_class]
        
        # Filter abilities by level
        available_abilities = []
        
        for ability_id, ability_data in class_abilities.items():
            if ability_data.get("level", 1) <= player_level:
                available_abilities.append({
                    "name": ability_data.get("name"),
                    "type": ability_data.get("type"),
                    "cooldown": ability_data.get("cooldown", 0),
                    "resource": ability_data.get("resource"),
                    "cost": ability_data.get("cost", 0)
                })
        
        # Sort abilities based on optimal rotation
        # This is a simplified version - a real implementation would have proper rotations
        
        # Priority order: buffs first, then abilities by level (highest first)
        buffs = [a for a in available_abilities if a.get("type") == "buff"]
        abilities = [a for a in available_abilities if a.get("type") == "ability"]
        
        # Sort buffs by name (alphabetical)
        buffs.sort(key=lambda a: a.get("name", ""))
        
        # Sort abilities by cooldown (longest first - assuming higher level abilities have longer cooldowns)
        abilities.sort(key=lambda a: a.get("cooldown", 0), reverse=True)
        
        # Combine: buffs first, then abilities
        rotation = buffs + abilities
        
        return rotation
    
    def get_quest_objective_info(self, quest_title: str, objective_name: str) -> Optional[Dict]:
        """
        Get information about a quest objective
        
        Args:
            quest_title: Title of the quest
            objective_name: Name of the objective
        
        Returns:
            Optional[Dict]: Objective information or None if not found
        """
        # Check standard knowledge base
        if quest_title in self.quests:
            quest_data = self.quests[quest_title]
            
            # Find matching objective
            for objective in quest_data.get("objectives", []):
                if objective.get("description", "") == objective_name:
                    return objective
                
                # Try partial match
                if objective_name.lower() in objective.get("description", "").lower():
                    return objective
        
        # Check runtime knowledge
        if quest_title in self.runtime_knowledge["quests"]:
            quest_data = self.runtime_knowledge["quests"][quest_title]
            
            # Find matching objective
            for objective in quest_data.get("objectives", []):
                if objective.get("description", "") == objective_name:
                    return objective
                
                # Try partial match
                if objective_name.lower() in objective.get("description", "").lower():
                    return objective
        
        # If not found, try to infer objective type from name
        obj_type = self._infer_objective_type(objective_name)
        
        # Create a basic objective info
        if obj_type:
            return {
                "type": obj_type,
                "description": objective_name,
                "target": self._extract_target_from_objective(objective_name),
                "count": 1
            }
        
        return None
    
    def _infer_objective_type(self, objective_name: str) -> Optional[str]:
        """
        Infer the type of an objective from its name
        
        Args:
            objective_name: Name of the objective
        
        Returns:
            Optional[str]: Inferred objective type or None
        """
        objective_name = objective_name.lower()
        
        # Check for kill objectives
        kill_keywords = ["kill", "slay", "defeat", "destroy"]
        for keyword in kill_keywords:
            if keyword in objective_name:
                return "kill"
        
        # Check for collection objectives
        collect_keywords = ["collect", "gather", "find", "obtain", "bring", "retrieve"]
        for keyword in collect_keywords:
            if keyword in objective_name:
                return "collect"
        
        # Check for interaction objectives
        interact_keywords = ["speak", "talk", "meet", "activate", "use", "click", "interact"]
        for keyword in interact_keywords:
            if keyword in objective_name:
                return "interact"
        
        # Check for exploration objectives
        explore_keywords = ["explore", "discover", "investigate", "scout"]
        for keyword in explore_keywords:
            if keyword in objective_name:
                return "explore"
        
        # Default to generic objective
        return "interact"
    
    def _extract_target_from_objective(self, objective_name: str) -> str:
        """
        Extract the target name from an objective description
        
        Args:
            objective_name: Name of the objective
        
        Returns:
            str: Extracted target name
        """
        objective_name = objective_name.lower()
        
        # Try to find the most specific target name
        # This is a very simplified implementation
        
        # Check for patterns like "Kill X", "Collect Y", etc.
        patterns = [
            "kill ", "slay ", "defeat ", "destroy ",  # Kill patterns
            "collect ", "gather ", "find ", "obtain ", "bring ", "retrieve ",  # Collect patterns
            "speak to ", "talk to ", "meet with "  # Interact patterns
        ]
        
        for pattern in patterns:
            if pattern in objective_name:
                # Get the part after the pattern
                target = objective_name.split(pattern)[1].strip()
                
                # Clean up by removing anything after certain words
                end_markers = [" and", " or", " from", " in", " at", " for", " to"]
                for marker in end_markers:
                    if marker in target:
                        target = target.split(marker)[0].strip()
                
                return target
        
        # If no pattern matched, just return the whole name
        return objective_name
    
    def get_path(self, start: Tuple[float, float], end: Tuple[float, float], 
                zone: str = "") -> List[Tuple[float, float]]:
        """
        Get a path between two points
        
        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            zone: Zone name (optional)
        
        Returns:
            List[Tuple[float, float]]: List of waypoints
        """
        # Check if we have a predefined path between these points
        for path_id, path_data in self.paths.items():
            path_start = tuple(path_data.get("waypoints", [[0, 0]])[0])
            path_end = tuple(path_data.get("waypoints", [[0, 0]])[-1])
            
            # Check if this path matches our start and end
            start_distance = self._calculate_distance(start, path_start)
            end_distance = self._calculate_distance(end, path_end)
            
            # If the path approximately matches (within 50 units)
            if start_distance < 50 and end_distance < 50:
                return [tuple(wp) for wp in path_data.get("waypoints", [])]
        
        # Check runtime knowledge paths
        for path_id, path_data in self.runtime_knowledge["paths"].items():
            path_start = tuple(path_data.get("waypoints", [[0, 0]])[0])
            path_end = tuple(path_data.get("waypoints", [[0, 0]])[-1])
            
            # Check if this path matches our start and end
            start_distance = self._calculate_distance(start, path_start)
            end_distance = self._calculate_distance(end, path_end)
            
            # If the path approximately matches (within 50 units)
            if start_distance < 50 and end_distance < 50:
                return [tuple(wp) for wp in path_data.get("waypoints", [])]
        
        # If no predefined path, create a simple direct path
        # In a real implementation, this would use A* pathfinding with terrain data
        
        # For simplicity, just return a straight line with a few waypoints
        path = [start]
        
        # Calculate distance and direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Create waypoints along the path
        num_waypoints = max(2, int(distance / 50))  # One waypoint per 50 units
        
        for i in range(1, num_waypoints):
            t = i / num_waypoints
            waypoint = (
                start[0] + dx * t,
                start[1] + dy * t
            )
            path.append(waypoint)
        
        path.append(end)
        
        # Remember this path for future use
        path_id = f"path_{len(self.runtime_knowledge['paths']) + 1}"
        self.runtime_knowledge["paths"][path_id] = {
            "waypoints": [[wp[0], wp[1]] for wp in path],
            "zone": zone
        }
        
        return path
    
    def get_quest_giver(self, quest_title: str) -> Optional[Dict]:
        """
        Get information about a quest giver
        
        Args:
            quest_title: Title of the quest
        
        Returns:
            Optional[Dict]: Quest giver information or None if not found
        """
        # Check standard knowledge base
        if quest_title in self.quests:
            quest_data = self.quests[quest_title]
            quest_giver_id = quest_data.get("quest_giver", "")
            
            if quest_giver_id and quest_giver_id in self.npcs:
                return self.npcs[quest_giver_id]
        
        # Check runtime knowledge
        if quest_title in self.runtime_knowledge["quests"]:
            quest_data = self.runtime_knowledge["quests"][quest_title]
            quest_giver_id = quest_data.get("quest_giver", "")
            
            if quest_giver_id:
                if quest_giver_id in self.npcs:
                    return self.npcs[quest_giver_id]
                elif quest_giver_id in self.runtime_knowledge["npcs"]:
                    return self.runtime_knowledge["npcs"][quest_giver_id]
        
        return None
    
    def get_quest_turnin(self, quest_title: str) -> Optional[Dict]:
        """
        Get information about a quest turn-in NPC
        
        Args:
            quest_title: Title of the quest
        
        Returns:
            Optional[Dict]: Quest turn-in information or None if not found
        """
        # Check standard knowledge base
        if quest_title in self.quests:
            quest_data = self.quests[quest_title]
            turnin_id = quest_data.get("turn_in", "")
            
            if turnin_id and turnin_id in self.npcs:
                return self.npcs[turnin_id]
        
        # Check runtime knowledge
        if quest_title in self.runtime_knowledge["quests"]:
            quest_data = self.runtime_knowledge["quests"][quest_title]
            turnin_id = quest_data.get("turn_in", "")
            
            if turnin_id:
                if turnin_id in self.npcs:
                    return self.npcs[turnin_id]
                elif turnin_id in self.runtime_knowledge["npcs"]:
                    return self.runtime_knowledge["npcs"][turnin_id]
        
        # If no specific turn-in, return quest giver (most quests are turned in to the same NPC)
        return self.get_quest_giver(quest_title)
    
    def get_quest_givers(self, zone: str) -> List[Dict]:
        """
        Get all quest givers in a zone
        
        Args:
            zone: Zone name
        
        Returns:
            List[Dict]: List of quest giver information
        """
        quest_givers = []
        
        # Check standard knowledge base
        for npc_id, npc_data in self.npcs.items():
            if npc_data.get("location", "") == zone and npc_data.get("gives_quests", []):
                quest_givers.append(npc_data)
        
        # Check runtime knowledge
        for npc_id, npc_data in self.runtime_knowledge["npcs"].items():
            if npc_data.get("location", "") == zone and npc_data.get("gives_quests", []):
                # Check if this NPC is already in the list
                if not any(qg.get("id", "") == npc_id for qg in quest_givers):
                    quest_givers.append(npc_data)
        
        return quest_givers
    
    def get_nearest_vendor(self, zone: str, position: Tuple[float, float]) -> Optional[Dict]:
        """
        Get the nearest vendor in a zone
        
        Args:
            zone: Zone name
            position: Current position (x, y)
        
        Returns:
            Optional[Dict]: Nearest vendor information or None if not found
        """
        vendors = []
        
        # Check standard knowledge base
        for npc_id, npc_data in self.npcs.items():
            if npc_data.get("location", "") == zone and npc_data.get("type", "") == "vendor":
                vendors.append(npc_data)
        
        # Check runtime knowledge
        for npc_id, npc_data in self.runtime_knowledge["npcs"].items():
            if npc_data.get("location", "") == zone and npc_data.get("type", "") == "vendor":
                # Check if this NPC is already in the list
                if not any(v.get("id", "") == npc_id for v in vendors):
                    vendors.append(npc_data)
        
        if not vendors:
            return None
        
        # Find the nearest vendor
        nearest_vendor = None
        nearest_distance = float('inf')
        
        for vendor in vendors:
            vendor_position = vendor.get("position", [0, 0])
            distance = self._calculate_distance(position, tuple(vendor_position))
            
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_vendor = vendor
        
        return nearest_vendor
    
    def get_unexplored_areas(self, zone: str, explored_areas: Set[str] = None) -> List[Dict]:
        """
        Get unexplored areas in a zone
        
        Args:
            zone: Zone name
            explored_areas: Set of already explored area names
        
        Returns:
            List[Dict]: List of unexplored areas
        """
        if explored_areas is None:
            explored_areas = set()
        
        unexplored = []
        
        # Check standard knowledge base
        if zone in self.zones:
            zone_data = self.zones[zone]
            
            # Check points of interest
            for poi in zone_data.get("points_of_interest", []):
                poi_name = poi.get("name", "")
                
                if poi_name and poi_name not in explored_areas:
                    unexplored.append({
                        "id": poi_name.lower().replace(" ", "_"),
                        "name": poi_name,
                        "position": tuple(poi.get("position", [0, 0])),
                        "type": "poi"
                    })
            
            # Check quest hubs
            for hub in zone_data.get("quest_hubs", []):
                if hub not in explored_areas:
                    # Find a position for this hub
                    hub_position = [0, 0]
                    
                    # Check if we have any NPCs in this hub
                    for npc_id, npc_data in self.npcs.items():
                        if npc_data.get("location", "") == hub:
                            hub_position = npc_data.get("position", [0, 0])
                            break
                    
                    unexplored.append({
                        "id": hub,
                        "name": hub.replace("_", " ").title(),
                        "position": tuple(hub_position),
                        "type": "quest_hub"
                    })
        
        # If no unexplored areas found, return an empty list
        return unexplored
    
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
    
    def _create_default_instances_data(self, file_path: str) -> None:
    """
    Create default instances data file
    
    Args:
        file_path: Path to save the file
    """
    default_data = {
        "deadmines": {
            "name": "Deadmines",
            "level_range": [15, 25],
            "faction": "both",
            "bosses": [
                {
                    "name": "Rhahk'Zor",
                    "strategy": "Simple first boss. Tank and spank."
                },
                {
                    "name": "Sneed",
                    "strategy": "Kill the Shredder first, then Sneed will come out."
                },
                {
                    "name": "Gilnid",
                    "strategy": "Kill the adds first, then focus on the boss."
                },
                {
                    "name": "Mr. Smite",
                    "strategy": "He stuns at health thresholds and switches weapons. Save cooldowns for after stuns."
                },
                {
                    "name": "Captain Greenskin",
                    "strategy": "Tank adds away from the group, focus down the boss."
                },
                {
                    "name": "Edwin VanCleef",
                    "strategy": "He summons adds at low health. Save cooldowns for this phase."
                }
            ],
            "tips": [
                "Watch for patrols in the foundry area.",
                "Be careful of the goblin engineers, they can call for help.",
                "The shortcut through the side tunnel can save time.",
                "After Mr. Smite, you can jump down to skip trash."
            ]
        },
        "wailing_caverns": {
            "name": "Wailing Caverns",
            "level_range": [15, 25],
            "faction": "both",
            "bosses": [
                {
                    "name": "Lady Anacondra",
                    "strategy": "Interrupt her Sleep spell. Tank and spank otherwise."
                },
                {
                    "name": "Lord Cobrahn",
                    "strategy": "He will transform into a snake at low health. Increased damage, so save defensives."
                },
                {
                    "name": "Kresh",
                    "strategy": "Optional boss. Simple tank and spank fight in the water."
                },
                {
                    "name": "Lord Pythas",
                    "strategy": "Interrupt his healing spell. Otherwise tank and spank."
                },
                {
                    "name": "Skum",
                    "strategy": "Optional boss. Fights in water. Can charge, so stay close."
                },
                {
                    "name": "Lord Serpentis",
                    "strategy": "Focuses random targets. Healers be aware."
                },
                {
                    "name": "Verdan the Everliving",
                    "strategy": "Final boss. Can knock players back and stun. Tank against a wall."
                }
            ],
            "tips": [
                "The dungeon is a maze. Follow the marked path or you'll get lost.",
                "There are four Fanglord bosses that must be killed before awakening Naralex.",
                "Deviate Faerie Dragons can put everyone to sleep. Kill them quickly.",
                "After all bosses, talk to the Disciple to start the final event."
            ]
        }
    }
    
    self.instances = default_data
    
    with open(file_path, 'w') as f:
        json.dump(default_data, f, indent=4)
    
    self.logger.info(f"Created default instances data at {file_path}")

    def _create_default_emotes_data(self, file_path: str) -> None:
        """
        Create default emotes data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "wave": {"text": "You wave.", "text_target": "You wave at %s.", "social_impact": "friendly"},
            "smile": {"text": "You smile.", "text_target": "You smile at %s.", "social_impact": "friendly"},
            "laugh": {"text": "You laugh.", "text_target": "You laugh at %s.", "social_impact": "friendly"},
            "thank": {"text": "You thank everyone.", "text_target": "You thank %s.", "social_impact": "friendly"},
            "cheer": {"text": "You cheer!", "text_target": "You cheer at %s!", "social_impact": "friendly"},
            "dance": {"text": "You dance.", "text_target": "You dance with %s.", "social_impact": "friendly"},
            "greet": {"text": "You greet everyone warmly.", "text_target": "You greet %s warmly.", "social_impact": "friendly"},
            "bow": {"text": "You bow.", "text_target": "You bow before %s.", "social_impact": "respectful"},
            "applaud": {"text": "You applaud.", "text_target": "You applaud at %s.", "social_impact": "friendly"},
            "salute": {"text": "You salute.", "text_target": "You salute %s.", "social_impact": "respectful"}
        }
        
        self.emotes = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default emotes data at {file_path}") 

    def get_instance_info(self, instance_name: str) -> Optional[Dict]:
        """
        Get information about a dungeon/raid instance
        
        Args:
            instance_name: Name of the instance
        
        Returns:
            Optional[Dict]: Instance information or None if not found
        """
        # Normalize the instance name
        instance_name = instance_name.lower().replace(' ', '_')
        
        # Check standard knowledge base
        if hasattr(self, 'instances') and instance_name in self.instances:
            return self.instances[instance_name]
        
        # Check runtime knowledge
        if "instances" in self.runtime_knowledge and instance_name in self.runtime_knowledge["instances"]:
            return self.runtime_knowledge["instances"][instance_name]
        
        return None 

    def get_emote_info(self, emote_name: str) -> Optional[Dict]:
        """
        Get information about an emote
        
        Args:
            emote_name: Name of the emote
        
        Returns:
            Optional[Dict]: Emote information or None if not found
        """
        # Normalize the emote name
        emote_name = emote_name.lower()
        
        # Check standard knowledge base
        if hasattr(self, 'emotes') and emote_name in self.emotes:
            return self.emotes[emote_name]
        
        return None 

    def get_social_response(self, category: str, context: Dict = None) -> Optional[str]:
        """
        Get an appropriate social response for a given category
        
        Args:
            category: Response category
            context: Context information (optional)
        
        Returns:
            Optional[str]: Response text or None if not found
        """
        if not hasattr(self, 'social_responses'):
            return None
        
        if category not in self.social_responses:
            return None
        
        responses = self.social_responses[category]
        if not responses:
            return None
        
        # Select a random response
        response_template = random.choice(responses)
        
        # Apply context if provided
        if context:
            try:
                return response_template.format(**context)
            except KeyError:
                # If formatting fails, return the template as-is
                return response_template
        else:
            return response_template