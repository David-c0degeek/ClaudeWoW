"""
Combat Manager Module

This module handles combat-related decision making and planning using class-specific
combat modules.
"""

import logging
import math
import random
import importlib
from typing import Dict, List, Tuple, Any, Optional
import time

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.combat.base_combat_module import BaseCombatModule
from src.decision.combat.warrior_combat_module import WarriorCombatModule
from src.decision.combat.mage_combat_module import MageCombatModule
from src.decision.combat.priest_combat_module import PriestCombatModule


class CombatManager:
    """
    Manages combat decisions and tactics using class-specific combat modules
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the CombatManager
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.combat_manager")
        self.config = config
        self.knowledge = knowledge
        
        # Combat module cache
        self.combat_modules: Dict[str, BaseCombatModule] = {}
        
        # Current combat state
        self.current_target = None
        self.last_cast_time = 0
        
        self.logger.info("CombatManager initialized")
    
    def generate_combat_plan(self, state: GameState) -> List[Dict]:
        """
        Generate a combat plan based on the current game state
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Combat plan steps
        """
        # Get the appropriate combat module for the player's class
        combat_module = self._get_combat_module(state)
        
        # If we have a valid combat module, use it to generate the plan
        if combat_module:
            return combat_module.generate_combat_plan(state)
        
        # Fallback if no module is available
        return self._generate_fallback_plan(state)
    
    def _get_combat_module(self, state: GameState) -> Optional[BaseCombatModule]:
        """
        Get the appropriate combat module for the player's class
        
        Args:
            state: Current game state
            
        Returns:
            Optional[BaseCombatModule]: Combat module or None
        """
        # Extract player class from state
        player_class = state.player_class.lower() if hasattr(state, "player_class") and state.player_class else "warrior"
        
        # Check if module is already cached
        if player_class in self.combat_modules:
            return self.combat_modules[player_class]
        
        # Try to import the appropriate module
        combat_module = None
        try:
            if player_class == "warrior":
                combat_module = WarriorCombatModule(self.config, self.knowledge)
            elif player_class == "mage":
                combat_module = MageCombatModule(self.config, self.knowledge)
            elif player_class == "priest":
                combat_module = PriestCombatModule(self.config, self.knowledge)
            else:
                # Try dynamic import based on class name
                module_name = f"{player_class}_combat_module"
                class_name = f"{player_class.capitalize()}CombatModule"
                
                try:
                    # Try to import dynamically
                    module = importlib.import_module(f"src.decision.combat.{module_name}")
                    module_class = getattr(module, class_name)
                    combat_module = module_class(self.config, self.knowledge)
                    self.logger.info(f"Dynamically loaded combat module for {player_class}")
                except (ImportError, AttributeError):
                    self.logger.warning(f"No combat module available for {player_class}, using fallback")
                    return None
        except Exception as e:
            self.logger.error(f"Error loading combat module for {player_class}: {e}")
            return None
        
        # Cache the module
        if combat_module:
            self.combat_modules[player_class] = combat_module
            self.logger.info(f"Loaded combat module for {player_class}")
        
        return combat_module
    
    def _generate_fallback_plan(self, state: GameState) -> List[Dict]:
        """
        Generate a basic fallback plan when no class-specific module is available
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Basic combat plan
        """
        plan = []
        
        try:
            # Check if we're in combat
            if not state.is_in_combat and not state.target:
                # Find a suitable target if not in combat
                suitable_target = self._find_suitable_target(state)
                
                if suitable_target:
                    # Add target selection to plan
                    plan.append({
                        "type": "target",
                        "target": suitable_target.get("id"),
                        "description": f"Target {suitable_target.get('id')}"
                    })
                else:
                    self.logger.info("No suitable targets found for combat")
                    return []
            
            # Get target info
            target = state.target if hasattr(state, "target") and state.target else suitable_target.get("id")
            
            # Get player class
            player_class = state.player_class.lower() if hasattr(state, "player_class") and state.player_class else "warrior"
            
            # Simple fallback abilities
            if player_class == "warrior":
                abilities = ["Attack", "Heroic Strike", "Rend"]
            elif player_class == "mage":
                abilities = ["Fireball", "Frostbolt", "Fire Blast"]
            elif player_class == "priest":
                abilities = ["Shadow Word: Pain", "Smite", "Mind Blast"]
            elif player_class == "hunter":
                abilities = ["Hunter's Mark", "Arcane Shot", "Serpent Sting"]
            else:
                abilities = ["Attack", "Special Ability"]
            
            # Add basic attack
            for ability in abilities:
                plan.append({
                    "type": "cast",
                    "spell": ability,
                    "target": target,
                    "description": f"Cast {ability} on {target}"
                })
        
        except Exception as e:
            self.logger.error(f"Error generating fallback combat plan: {e}")
            # Ultimate fallback to basic attack
            plan = [{
                "type": "cast",
                "spell": "Attack",
                "target": state.target if hasattr(state, "target") and state.target else "target",
                "description": "Basic attack"
            }]
        
        return plan
    
    def _find_suitable_target(self, state: GameState) -> Optional[Dict]:
        """
        Find a suitable combat target (used by fallback plan)
        
        Args:
            state: Current game state
        
        Returns:
            Optional[Dict]: Target entity or None
        """
        # Check if we have active quests that require killing specific enemies
        quest_targets = []
        if hasattr(state, "active_quests") and state.active_quests:
            for quest in state.active_quests:
                objectives = quest.get("objectives", [])
                for objective in objectives:
                    obj_name = objective.get("name", "")
                    current = objective.get("current", 0)
                    total = objective.get("total", 1)
                    
                    if current < total:
                        # Check if this is a kill objective
                        obj_info = self.knowledge.get_quest_objective_info(
                            quest.get("title", ""), obj_name)
                        
                        if obj_info and obj_info.get("type") == "kill":
                            target_name = obj_info.get("target", "")
                            quest_targets.append(target_name)
        
        # Filter nearby entities for quest targets first
        suitable_targets = []
        
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                entity_name = entity.get("id", "")
                entity_type = entity.get("type", "")
                entity_reaction = entity.get("reaction", "")
                
                # Check if it's a quest target
                is_quest_target = any(qt.lower() in entity_name.lower() for qt in quest_targets)
                
                # Check if it's a hostile entity
                is_hostile = entity_reaction == "hostile" or entity_type == "mob"
                
                if is_quest_target or is_hostile:
                    suitable_targets.append(entity)
            
            # If we found suitable targets, sort by priority
            if suitable_targets:
                # Prioritize quest targets over regular hostile mobs
                suitable_targets.sort(key=lambda e: (
                    not any(qt.lower() in e.get("id", "").lower() for qt in quest_targets),
                    # Then sort by level (prefer same level or slightly lower)
                    abs(e.get("level", state.player_level) - state.player_level) 
                    if hasattr(state, "player_level") and state.player_level else 0
                ))
                
                return suitable_targets[0]
        
        return None
    
    def get_supported_classes(self) -> List[str]:
        """
        Get a list of classes with dedicated combat modules
        
        Returns:
            List[str]: List of supported class names
        """
        # Return the list of supported classes
        return ["Warrior", "Mage", "Priest"]
    
    def get_talent_builds_for_class(self, class_name: str) -> List[Dict[str, Any]]:
        """
        Get a list of supported talent builds for a given class
        
        Args:
            class_name: Name of the class
            
        Returns:
            List[Dict]: List of supported talent builds
        """
        # Get the combat module for this class
        state = GameState()
        state.player_class = class_name
        combat_module = self._get_combat_module(state)
        
        # Return the supported talent builds
        if combat_module:
            return combat_module.get_supported_talent_builds()
        
        return []