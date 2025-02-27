"""
Planner Module

This module handles high-level planning and goal decomposition.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import heapq
import time

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge

class Planner:
    """
    Plans and decomposes high-level goals into executable actions
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the Planner
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.planner")
        self.config = config
        self.knowledge = knowledge
        
        # Planning parameters
        self.planning_horizon = config.get("planning_horizon", 10)  # Steps to plan ahead
        self.max_planning_time = config.get("max_planning_time", 0.1)  # Max planning time in seconds
        
        self.logger.info("Planner initialized")
    
    def create_plan(self, goal: Dict, state: GameState) -> List[Dict]:
        """
        Create a plan to achieve a high-level goal
        
        Args:
            goal: Goal description
            state: Current game state
        
        Returns:
            List[Dict]: Sequence of plan steps
        """
        self.logger.info(f"Creating plan for goal: {goal.get('type')}")
        
        # Start planning timer
        start_time = time.time()
        
        # Select planning method based on goal type
        goal_type = goal.get("type", "")
        
        if goal_type == "combat":
            plan = self._plan_combat(goal, state)
        elif goal_type == "quest":
            plan = self._plan_quest(goal, state)
        elif goal_type == "navigation":
            plan = self._plan_navigation(goal, state)
        elif goal_type == "loot":
            plan = self._plan_loot(goal, state)
        elif goal_type == "vendor":
            plan = self._plan_vendor(goal, state)
        elif goal_type == "exploration":
            plan = self._plan_exploration(goal, state)
        else:
            self.logger.warning(f"Unknown goal type: {goal_type}")
            plan = []
        
        # Check planning time
        planning_time = time.time() - start_time
        if planning_time > self.max_planning_time:
            self.logger.warning(f"Planning took {planning_time:.3f}s, exceeding limit of {self.max_planning_time}s")
        
        return plan
    
    def _plan_combat(self, goal: Dict, state: GameState) -> List[Dict]:
        """
        Plan for combat goal
        
        Args:
            goal: Combat goal description
            state: Current game state
        
        Returns:
            List[Dict]: Combat plan steps
        """
        plan = []
        target = goal.get("target")
        
        # If no target specified, find an appropriate one
        if not target and state.nearby_entities:
            # Find hostile entities
            hostile_entities = [e for e in state.nearby_entities 
                              if e.get("reaction") == "hostile"]
            
            if hostile_entities:
                target = hostile_entities[0].get("id")
                
        # If we have a target, plan the combat sequence
        if target:
            # Define player class and level
            player_class = state.player_class if state.player_class else "warrior"  # Default to warrior
            player_level = state.player_level if state.player_level else 1
            
            # Get combat rotation from knowledge base
            rotation = self.knowledge.get_combat_rotation(player_class, player_level)
            
            # Add target selection step
            plan.append({
                "type": "target",
                "target": target,
                "description": f"Target {target}"
            })
            
            # Add positioning step
            plan.append({
                "type": "move",
                "position": "combat_range",  # Special value indicating proper combat range
                "description": "Move to combat range"
            })
            
            # Add combat steps based on rotation
            for ability in rotation:
                plan.append({
                    "type": "cast",
                    "spell": ability.get("name"),
                    "target": target,
                    "description": f"Cast {ability.get('name')}"
                })
        
        return plan
    
    def _plan_quest(self, goal: Dict, state: GameState) -> List[Dict]:
        """
        Plan for quest goal
        
        Args:
            goal: Quest goal description
            state: Current game state
        
        Returns:
            List[Dict]: Quest plan steps
        """
        plan = []
        quest = goal.get("quest", {})
        
        if not quest:
            # No specific quest, check for any active quests
            if state.active_quests:
                quest = state.active_quests[0]
        
        if quest:
            quest_title = quest.get("title", "Unknown Quest")
            self.logger.info(f"Planning for quest: {quest_title}")
            
            # Check quest objectives
            objectives = quest.get("objectives", [])
            
            for objective in objectives:
                obj_name = objective.get("name", "")
                current = objective.get("current", 0)
                total = objective.get("total", 1)
                
                # If objective not complete, plan for it
                if current < total:
                    # Query knowledge base for objective information
                    obj_info = self.knowledge.get_quest_objective_info(quest_title, obj_name)
                    
                    if obj_info:
                        # Add navigation to objective area
                        if "location" in obj_info:
                            plan.append({
                                "type": "navigate",
                                "destination": obj_info["location"],
                                "description": f"Navigate to {obj_name} area"
                            })
                        
                        # Add steps based on objective type
                        if obj_info.get("type") == "kill":
                            # Add steps to kill target mobs
                            mob_name = obj_info.get("target", "")
                            plan.append({
                                "type": "find",
                                "target": mob_name,
                                "description": f"Find {mob_name}"
                            })
                            plan.append({
                                "type": "combat",
                                "target": mob_name,
                                "description": f"Kill {mob_name}"
                            })
                        
                        elif obj_info.get("type") == "collect":
                            # Add steps to collect items
                            item_name = obj_info.get("item", "")
                            plan.append({
                                "type": "find",
                                "target": item_name,
                                "description": f"Find {item_name}"
                            })
                            plan.append({
                                "type": "interact",
                                "target": item_name,
                                "description": f"Collect {item_name}"
                            })
                        
                        elif obj_info.get("type") == "interact":
                            # Add steps to interact with objects/NPCs
                            interact_target = obj_info.get("target", "")
                            plan.append({
                                "type": "find",
                                "target": interact_target,
                                "description": f"Find {interact_target}"
                            })
                            plan.append({
                                "type": "interact",
                                "target": interact_target,
                                "description": f"Interact with {interact_target}"
                            })
            
            # If no incomplete objectives found, plan to turn in quest
            if not plan and quest.get("can_complete", False):
                # Query knowledge base for quest giver
                quest_giver = self.knowledge.get_quest_giver(quest_title)
                
                if quest_giver:
                    plan.append({
                        "type": "navigate",
                        "destination": quest_giver.get("location", ""),
                        "description": f"Navigate to {quest_giver.get('name', 'quest giver')}"
                    })
                    plan.append({
                        "type": "interact",
                        "target": quest_giver.get("id", ""),
                        "description": f"Turn in quest to {quest_giver.get('name', 'quest giver')}"
                    })
        
        return plan
    
    def _plan_navigation(self, goal: Dict, state: GameState) -> List[Dict]:
        """
        Plan for navigation goal
        
        Args:
            goal: Navigation goal description
            state: Current game state
        
        Returns:
            List[Dict]: Navigation plan steps
        """
        plan = []
        destination = goal.get("destination")
        
        if destination:
            # Get current position
            current_pos = state.player_position
            
            # Query knowledge base for path
            path = self.knowledge.get_path(current_pos, destination)
            
            if path:
                # Convert path to navigation steps
                for i, waypoint in enumerate(path):
                    plan.append({
                        "type": "move",
                        "position": waypoint,
                        "description": f"Move to waypoint {i+1}/{len(path)}"
                    })
            else:
                # Fallback to direct movement
                plan.append({
                    "type": "move",
                    "position": destination,
                    "description": f"Move directly to destination"
                })
        
        return plan
    
    def _plan_loot(self, goal: Dict, state: GameState) -> List[Dict]:
        """
        Plan for looting
        
        Args:
            goal: Loot goal description
            state: Current game state
        
        Returns:
            List[Dict]: Loot plan steps
        """
        plan = []
        
        # Find lootable entities
        lootable_entities = [e for e in state.nearby_entities 
                           if e.get("type") == "lootable"]
        
        # Sort by distance (closest first)
        if hasattr(state, "player_position") and state.player_position:
            lootable_entities.sort(key=lambda e: self._calculate_distance(
                state.player_position, e.get("position", (0, 0))))
        
        # Plan to loot each entity
        for entity in lootable_entities:
            plan.append({
                "type": "move",
                "position": entity.get("position"),
                "description": f"Move to loot {entity.get('id', 'corpse')}"
            })
            plan.append({
                "type": "interact",
                "target": entity.get("id"),
                "description": f"Loot {entity.get('id', 'corpse')}"
            })
        
        return plan
    
    def _plan_vendor(self, goal: Dict, state: GameState) -> List[Dict]:
        """
        Plan for vendor visit
        
        Args:
            goal: Vendor goal description
            state: Current game state
        
        Returns:
            List[Dict]: Vendor plan steps
        """
        plan = []
        
        # Check if vendors are nearby
        vendor_entities = [e for e in state.nearby_entities 
                         if e.get("subtype") == "vendor"]
        
        if vendor_entities:
            # Use nearby vendor
            vendor = vendor_entities[0]
            
            plan.append({
                "type": "move",
                "position": vendor.get("position"),
                "description": f"Move to vendor {vendor.get('id', 'vendor')}"
            })
            plan.append({
                "type": "interact",
                "target": vendor.get("id"),
                "description": f"Interact with vendor {vendor.get('id', 'vendor')}"
            })
            plan.append({
                "type": "sell_items",
                "items": ["junk", "gray_items"],
                "description": "Sell junk items"
            })
            plan.append({
                "type": "repair",
                "description": "Repair gear"
            })
        else:
            # Find nearest vendor from knowledge base
            current_zone = state.current_zone
            vendor_info = self.knowledge.get_nearest_vendor(
                current_zone, state.player_position)
            
            if vendor_info:
                plan.append({
                    "type": "navigate",
                    "destination": vendor_info.get("location"),
                    "description": f"Navigate to vendor {vendor_info.get('name', 'vendor')}"
                })
                plan.append({
                    "type": "interact",
                    "target": vendor_info.get("id"),
                    "description": f"Interact with vendor {vendor_info.get('name', 'vendor')}"
                })
                plan.append({
                    "type": "sell_items",
                    "items": ["junk", "gray_items"],
                    "description": "Sell junk items"
                })
                plan.append({
                    "type": "repair",
                    "description": "Repair gear"
                })
        
        return plan
    
    def _plan_exploration(self, goal: Dict, state: GameState) -> List[Dict]:
        """
        Plan for exploration
        
        Args:
            goal: Exploration goal description
            state: Current game state
        
        Returns:
            List[Dict]: Exploration plan steps
        """
        plan = []
        
        # Query knowledge base for unexplored areas
        current_zone = state.current_zone
        unexplored_areas = self.knowledge.get_unexplored_areas(current_zone)
        
        if unexplored_areas:
            # Pick an unexplored area to explore
            target_area = unexplored_areas[0]
            
            plan.append({
                "type": "navigate",
                "destination": target_area.get("location"),
                "description": f"Explore {target_area.get('name', 'area')}"
            })
        else:
            # If no specific unexplored areas, just move in a random direction
            plan.append({
                "type": "random_move",
                "duration": 10,
                "description": "Explore surroundings"
            })
        
        return plan
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate distance between two positions
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
        
        Returns:
            float: Distance between positions
        """
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5