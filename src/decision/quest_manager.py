"""
Quest Manager Module

This module handles quest-related decision making and planning.
"""

import logging
import random
from typing import Dict, List, Tuple, Any, Optional
import math
import time

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge

class QuestManager:
    """
    Manages quest-related decisions and planning
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the QuestManager
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.quest_manager")
        self.config = config
        self.knowledge = knowledge
        
        # Quest state tracking
        self.active_quests = {}  # quest_id -> quest_info
        self.completed_quests = set()
        self.quest_progress = {}  # quest_id -> objective_progress
        
        # Quest preference settings
        self.quest_level_range = config.get("quest_level_range", 5)  # Accept quests within this level range
        self.max_active_quests = config.get("max_active_quests", 25)  # Maximum number of active quests
        self.prioritize_low_level_quests = config.get("prioritize_low_level_quests", True)  # Finish low-level quests first
        
        self.logger.info("QuestManager initialized")
    
    def generate_quest_plan(self, state: GameState) -> List[Dict]:
        """
        Generate a plan for quest progression
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Quest plan steps
        """
        plan = []
        
        try:
            # Update quest state from game state
            self._update_quest_state(state)
            
            # Check if we have any active quests
            if not state.active_quests:
                # No active quests, look for available quests nearby
                return self._generate_quest_acquisition_plan(state)
            
            # Determine which quest to work on
            target_quest = self._select_target_quest(state)
            
            if not target_quest:
                self.logger.warning("No suitable quest found to work on")
                return []
            
            quest_title = target_quest.get("title", "Unknown Quest")
            self.logger.info(f"Selected quest to work on: {quest_title}")
            
            # Check if quest can be turned in
            if target_quest.get("can_complete", False):
                # Generate plan to turn in the quest
                return self._generate_quest_turnin_plan(state, target_quest)
            
            # Generate plan to work on quest objectives
            objectives = target_quest.get("objectives", [])
            incomplete_objectives = [
                obj for obj in objectives
                if obj.get("current", 0) < obj.get("total", 1)
            ]
            
            if incomplete_objectives:
                # Work on the first incomplete objective
                objective = incomplete_objectives[0]
                objective_plan = self._generate_objective_plan(state, target_quest, objective)
                plan.extend(objective_plan)
            else:
                # No incomplete objectives but quest not marked as completable
                # This might happen if the quest completion state is not properly detected
                self.logger.warning(f"No incomplete objectives for {quest_title} but quest not marked as completable")
                # Try to generate turn-in plan anyway
                plan = self._generate_quest_turnin_plan(state, target_quest)
        
        except Exception as e:
            self.logger.error(f"Error generating quest plan: {e}")
            # Fallback to simple plan
            plan = [{
                "type": "wait",
                "duration": 2.0,
                "description": "Pausing quest activities due to error"
            }]
        
        return plan
    
    def _update_quest_state(self, state: GameState) -> None:
        """
        Update internal quest state from game state
        
        Args:
            state: Current game state
        """
        # Update active quests
        if hasattr(state, "active_quests") and state.active_quests:
            for quest in state.active_quests:
                quest_id = quest.get("title", "")
                
                if quest_id and quest_id not in self.completed_quests:
                    # Update or add quest to active quests
                    self.active_quests[quest_id] = quest
                    
                    # Update quest progress
                    if quest_id not in self.quest_progress:
                        self.quest_progress[quest_id] = {}
                    
                    for objective in quest.get("objectives", []):
                        obj_name = objective.get("name", "")
                        if obj_name:
                            self.quest_progress[quest_id][obj_name] = {
                                "current": objective.get("current", 0),
                                "total": objective.get("total", 1),
                                "last_updated": time.time()
                            }
    
    def _select_target_quest(self, state: GameState) -> Optional[Dict]:
        """
        Select a quest to work on
        
        Args:
            state: Current game state
        
        Returns:
            Optional[Dict]: Selected quest or None
        """
        if not hasattr(state, "active_quests") or not state.active_quests:
            return None
        
        active_quests = state.active_quests
        
        # Check for completable quests first
        completable_quests = [q for q in active_quests if q.get("can_complete", False)]
        if completable_quests:
            return completable_quests[0]
        
        # Check for nearby quest objectives
        player_position = state.player_position if hasattr(state, "player_position") else None
        
        if player_position:
            # Filter quests by proximity to objectives
            nearby_quests = []
            
            for quest in active_quests:
                quest_title = quest.get("title", "")
                objectives = quest.get("objectives", [])
                
                for objective in objectives:
                    obj_name = objective.get("name", "")
                    current = objective.get("current", 0)
                    total = objective.get("total", 1)
                    
                    if current < total:
                        # Check if this objective is nearby
                        obj_info = self.knowledge.get_quest_objective_info(quest_title, obj_name)
                        
                        if obj_info and "location" in obj_info:
                            obj_location = obj_info["location"]
                            distance = self._calculate_distance(player_position, obj_location)
                            
                            # If objective is nearby, prioritize this quest
                            if distance < 200:  # Arbitrary distance threshold
                                nearby_quests.append((quest, distance))
            
            # Sort by distance and return closest
            if nearby_quests:
                nearby_quests.sort(key=lambda q: q[1])
                return nearby_quests[0][0]
        
        # If no nearby quests, prioritize by level or completion
        if self.prioritize_low_level_quests:
            # Sort by level (lowest first)
            sorted_quests = sorted(active_quests, key=lambda q: q.get("level", 0))
        else:
            # Sort by completion percentage (highest first)
            sorted_quests = sorted(active_quests, key=lambda q: self._calculate_completion_percentage(q), reverse=True)
        
        return sorted_quests[0] if sorted_quests else None
    
    def _generate_quest_acquisition_plan(self, state: GameState) -> List[Dict]:
        """
        Generate a plan to find and accept new quests
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        
        # Check minimap for quest givers
        quest_givers = []
        
        if hasattr(state, "minimap_data") and state.minimap_data:
            quest_markers = state.minimap_data.get("quest_markers", [])
            for marker in quest_markers:
                if marker.get("type") == "available_quest":
                    quest_givers.append({
                        "position": marker.get("position"),
                        "id": marker.get("id", "quest_giver")
                    })
        
        # Check nearby entities for quest givers
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                # Check if entity has quest marker
                if entity.get("subtype") == "quest_giver":
                    quest_givers.append({
                        "position": entity.get("position"),
                        "id": entity.get("id")
                    })
        
        # If no quest givers found, use knowledge base
        if not quest_givers:
            player_position = state.player_position if hasattr(state, "player_position") else None
            current_zone = state.current_zone if hasattr(state, "current_zone") else ""
            
            if player_position and current_zone:
                known_quest_givers = self.knowledge.get_quest_givers(current_zone)
                
                for quest_giver in known_quest_givers:
                    quest_givers.append({
                        "position": quest_giver.get("position"),
                        "id": quest_giver.get("id"),
                        "name": quest_giver.get("name")
                    })
        
        # Sort quest givers by distance
        player_position = state.player_position if hasattr(state, "player_position") else None
        
        if player_position and quest_givers:
            quest_givers.sort(key=lambda qg: self._calculate_distance(
                player_position, qg.get("position", (0, 0))))
        
        # Generate plan for the closest quest giver
        if quest_givers:
            quest_giver = quest_givers[0]
            
            # Add movement to quest giver
            plan.append({
                "type": "move",
                "position": quest_giver.get("position"),
                "description": f"Move to quest giver {quest_giver.get('name', quest_giver.get('id', 'NPC'))}"
            })
            
            # Add interaction with quest giver
            plan.append({
                "type": "interact",
                "target": quest_giver.get("id"),
                "description": f"Talk to quest giver {quest_giver.get('name', quest_giver.get('id', 'NPC'))}"
            })
            
            # Add accepting the quest
            plan.append({
                "type": "accept_quest",
                "description": "Accept quest"
            })
        else:
            # No quest givers found, explore to find some
            plan.append({
                "type": "explore",
                "duration": 60,
                "description": "Explore to find quest givers"
            })
        
        return plan
    
    def _generate_quest_turnin_plan(self, state: GameState, quest: Dict) -> List[Dict]:
        """
        Generate a plan to turn in a completed quest
        
        Args:
            state: Current game state
            quest: Quest information
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        quest_title = quest.get("title", "Unknown Quest")
        
        # Get quest giver info for turn-in
        quest_giver = self.knowledge.get_quest_turnin(quest_title)
        
        if not quest_giver:
            # If no specific turn-in NPC found, try the original quest giver
            quest_giver = self.knowledge.get_quest_giver(quest_title)
        
        if quest_giver:
            # Add movement to quest turn-in NPC
            plan.append({
                "type": "navigate",
                "destination": quest_giver.get("position", quest_giver.get("location")),
                "description": f"Navigate to {quest_giver.get('name', 'quest turn-in NPC')}"
            })
            
            # Add interaction with quest NPC
            plan.append({
                "type": "interact",
                "target": quest_giver.get("id"),
                "description": f"Talk to {quest_giver.get('name', 'quest turn-in NPC')}"
            })
            
            # Add turning in the quest
            plan.append({
                "type": "turnin_quest",
                "quest": quest_title,
                "description": f"Turn in quest: {quest_title}"
            })
            
            # Choose quest reward if applicable
            plan.append({
                "type": "select_reward",
                "strategy": "best_upgrade",
                "description": "Select best quest reward"
            })
        else:
            self.logger.warning(f"No turn-in NPC found for quest: {quest_title}")
            
            # Check minimap for turn-in markers
            if hasattr(state, "minimap_data") and state.minimap_data:
                quest_markers = state.minimap_data.get("quest_markers", [])
                
                # Filter for turn-in markers (yellow ?)
                turnin_markers = [m for m in quest_markers if m.get("type") == "quest_complete"]
                
                if turnin_markers:
                    marker = turnin_markers[0]
                    
                    # Add movement to marker
                    plan.append({
                        "type": "move",
                        "position": marker.get("position"),
                        "description": "Move to quest turn-in marker"
                    })
                    
                    # Add interaction
                    plan.append({
                        "type": "interact",
                        "target": marker.get("id", "quest_turnin"),
                        "description": "Interact with quest turn-in NPC"
                    })
                    
                    # Add turning in the quest
                    plan.append({
                        "type": "turnin_quest",
                        "quest": quest_title,
                        "description": f"Turn in quest: {quest_title}"
                    })
                    
                    # Choose quest reward if applicable
                    plan.append({
                        "type": "select_reward",
                        "strategy": "best_upgrade",
                        "description": "Select best quest reward"
                    })
                else:
                    # No markers found, generic exploration
                    plan.append({
                        "type": "explore",
                        "duration": 60,
                        "description": "Explore to find quest turn-in NPC"
                    })
        
        return plan
    
    def _generate_objective_plan(self, state: GameState, quest: Dict, objective: Dict) -> List[Dict]:
        """
        Generate a plan to complete a quest objective
        
        Args:
            state: Current game state
            quest: Quest information
            objective: Objective information
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        quest_title = quest.get("title", "Unknown Quest")
        obj_name = objective.get("name", "")
        
        # Get objective info from knowledge base
        obj_info = self.knowledge.get_quest_objective_info(quest_title, obj_name)
        
        if not obj_info:
            self.logger.warning(f"No information found for objective: {obj_name} in quest: {quest_title}")
            return []
        
        # Get objective type
        obj_type = obj_info.get("type", "unknown")
        
        if obj_type == "kill":
            # Plan for kill objective
            return self._generate_kill_objective_plan(state, quest, objective, obj_info)
        
        elif obj_type == "collect":
            # Plan for collection objective
            return self._generate_collect_objective_plan(state, quest, objective, obj_info)
        
        elif obj_type == "interact":
            # Plan for interaction objective
            return self._generate_interact_objective_plan(state, quest, objective, obj_info)
        
        elif obj_type == "explore":
            # Plan for exploration objective
            return self._generate_explore_objective_plan(state, quest, objective, obj_info)
        
        else:
            # Unknown objective type, generic approach
            self.logger.warning(f"Unknown objective type: {obj_type} for {obj_name}")
            
            # If we have a location, at least go there
            if "location" in obj_info:
                plan.append({
                    "type": "navigate",
                    "destination": obj_info["location"],
                    "description": f"Navigate to objective area for {obj_name}"
                })
                
                # Generic exploration at location
                plan.append({
                    "type": "explore",
                    "duration": 30,
                    "description": f"Explore area to find way to complete {obj_name}"
                })
        
        return plan
    
    def _generate_kill_objective_plan(self, state: GameState, quest: Dict, 
                                    objective: Dict, obj_info: Dict) -> List[Dict]:
        """
        Generate a plan for a kill objective
        
        Args:
            state: Current game state
            quest: Quest information
            objective: Objective information
            obj_info: Additional objective info from knowledge base
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        obj_name = objective.get("name", "")
        target_mob = obj_info.get("target", "")
        
        # Add navigation to mob area if location is known
        if "location" in obj_info:
            plan.append({
                "type": "navigate",
                "destination": obj_info["location"],
                "description": f"Navigate to {target_mob} area"
            })
        
        # Check if target mobs are already nearby
        target_nearby = False
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                entity_name = entity.get("id", "").lower()
                
                # Check if this entity matches the target
                if target_mob.lower() in entity_name:
                    target_nearby = True
                    
                    # Add movement to target
                    plan.append({
                        "type": "move",
                        "position": entity.get("position"),
                        "description": f"Move to {target_mob}"
                    })
                    
                    # Add target selection
                    plan.append({
                        "type": "target",
                        "target": entity.get("id"),
                        "description": f"Target {target_mob}"
                    })
                    
                    # Add combat sequence
                    plan.append({
                        "type": "combat",
                        "target": entity.get("id"),
                        "description": f"Kill {target_mob}"
                    })
                    
                    # Add looting
                    plan.append({
                        "type": "loot",
                        "target": entity.get("id"),
                        "description": f"Loot {target_mob}"
                    })
                    
                    break
        
        if not target_nearby:
            # If no targets nearby, add search action
            plan.append({
                "type": "search",
                "target": target_mob,
                "duration": 30,
                "description": f"Search for {target_mob}"
            })
        
        return plan
    
    def _generate_collect_objective_plan(self, state: GameState, quest: Dict, 
                                       objective: Dict, obj_info: Dict) -> List[Dict]:
        """
        Generate a plan for a collection objective
        
        Args:
            state: Current game state
            quest: Quest information
            objective: Objective information
            obj_info: Additional objective info from knowledge base
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        obj_name = objective.get("name", "")
        item_name = obj_info.get("item", "")
        
        # Add navigation to item area if location is known
        if "location" in obj_info:
            plan.append({
                "type": "navigate",
                "destination": obj_info["location"],
                "description": f"Navigate to {item_name} area"
            })
        
        # Check if we need to kill mobs to get the item
        if "source" in obj_info and obj_info["source"] == "mob":
            mob_name = obj_info.get("mob", "")
            
            # Add mob hunting steps
            plan.append({
                "type": "search",
                "target": mob_name,
                "duration": 30,
                "description": f"Search for {mob_name} that drop {item_name}"
            })
            
            # Add combat sequence
            plan.append({
                "type": "combat",
                "target": mob_name,
                "description": f"Kill {mob_name}"
            })
            
            # Add looting
            plan.append({
                "type": "loot",
                "target": mob_name,
                "description": f"Loot {item_name} from {mob_name}"
            })
        
        else:
            # Item is gathered from the world
            
            # Check if item nodes are already nearby
            item_nearby = False
            if hasattr(state, "nearby_entities") and state.nearby_entities:
                for entity in state.nearby_entities:
                    entity_name = entity.get("id", "").lower()
                    
                    # Check if this entity matches the item
                    if item_name.lower() in entity_name:
                        item_nearby = True
                        
                        # Add movement to item
                        plan.append({
                            "type": "move",
                            "position": entity.get("position"),
                            "description": f"Move to {item_name}"
                        })
                        
                        # Add interaction
                        plan.append({
                            "type": "interact",
                            "target": entity.get("id"),
                            "description": f"Collect {item_name}"
                        })
                        
                        break
            
            if not item_nearby:
                # If no items nearby, add search action
                plan.append({
                    "type": "search",
                    "target": item_name,
                    "duration": 30,
                    "description": f"Search for {item_name}"
                })
        
        return plan
    
    def _generate_interact_objective_plan(self, state: GameState, quest: Dict, 
                                        objective: Dict, obj_info: Dict) -> List[Dict]:
        """
        Generate a plan for an interaction objective
        
        Args:
            state: Current game state
            quest: Quest information
            objective: Objective information
            obj_info: Additional objective info from knowledge base
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        obj_name = objective.get("name", "")
        interact_target = obj_info.get("target", "")
        
        # Add navigation to target area if location is known
        if "location" in obj_info:
            plan.append({
                "type": "navigate",
                "destination": obj_info["location"],
                "description": f"Navigate to {interact_target}"
            })
        
        # Check if target is already nearby
        target_nearby = False
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                entity_name = entity.get("id", "").lower()
                
                # Check if this entity matches the target
                if interact_target.lower() in entity_name:
                    target_nearby = True
                    
                    # Add movement to target
                    plan.append({
                        "type": "move",
                        "position": entity.get("position"),
                        "description": f"Move to {interact_target}"
                    })
                    
                    # Add interaction
                    plan.append({
                        "type": "interact",
                        "target": entity.get("id"),
                        "description": f"Interact with {interact_target}"
                    })
                    
                    break
        
        if not target_nearby:
            # If no target nearby, add search action
            plan.append({
                "type": "search",
                "target": interact_target,
                "duration": 30,
                "description": f"Search for {interact_target}"
            })
        
        return plan
    
    def _generate_explore_objective_plan(self, state: GameState, quest: Dict, 
                                       objective: Dict, obj_info: Dict) -> List[Dict]:
        """
        Generate a plan for an exploration objective
        
        Args:
            state: Current game state
            quest: Quest information
            objective: Objective information
            obj_info: Additional objective info from knowledge base
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        obj_name = objective.get("name", "")
        area_name = obj_info.get("area", "")
        
        # Add navigation to area if location is known
        if "location" in obj_info:
            plan.append({
                "type": "navigate",
                "destination": obj_info["location"],
                "description": f"Navigate to {area_name}"
            })
            
            # Add exploration
            plan.append({
                "type": "explore",
                "area": area_name,
                "duration": 60,
                "description": f"Explore {area_name}"
            })
        else:
            # If no specific location, generic exploration
            plan.append({
                "type": "explore",
                "duration": 60,
                "description": f"Explore to find {area_name}"
            })
        
        return plan
    
    def _calculate_completion_percentage(self, quest: Dict) -> float:
        """
        Calculate the completion percentage of a quest
        
        Args:
            quest: Quest information
        
        Returns:
            float: Completion percentage (0.0 to 100.0)
        """
        objectives = quest.get("objectives", [])
        if not objectives:
            return 0.0
        
        completed = 0
        total = 0
        
        for obj in objectives:
            current = obj.get("current", 0)
            max_value = obj.get("total", 1)
            
            completed += current
            total += max_value
        
        if total == 0:
            return 0.0
        
        return (completed / total) * 100.0
    
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