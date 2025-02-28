"""
Base Combat Module

This module serves as the foundation for all class-specific combat modules.
It defines the common interface and shared functionality that all combat
implementations will inherit.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Set

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge


class BaseCombatModule(ABC):
    """
    Base class for all class-specific combat modules.
    
    This abstract class defines the interface that all combat modules must implement,
    including methods for rotation management, resource tracking, and ability usage.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge: GameKnowledge):
        """
        Initialize the base combat module
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger(f"wow_ai.decision.combat.{self.__class__.__name__}")
        self.config = config
        self.knowledge = knowledge
        
        # Cooldown tracking
        self.ability_cooldowns: Dict[str, float] = {}  # Ability name -> timestamp when available
        self.global_cooldown: float = 0.0  # Timestamp when GCD ends
        
        # Resource tracking
        self.current_resources: Dict[str, float] = {}  # Resource type -> current value
        self.max_resources: Dict[str, float] = {}  # Resource type -> max value
        
        # Spell/ability history
        self.cast_history: List[Dict[str, Any]] = []  # Recent cast history
        self.last_cast_time: float = 0.0
        
        # Target tracking
        self.current_target: Optional[str] = None
        self.target_data: Dict[str, Any] = {}
        
        # Buff/debuff tracking
        self.active_buffs: Dict[str, Dict[str, Any]] = {}  # Buff name -> data
        self.target_debuffs: Dict[str, Dict[str, Any]] = {}  # Debuff name -> data
        
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    def update_state(self, state: GameState) -> None:
        """
        Update internal state based on current game state
        
        Args:
            state: Current game state
        """
        # Update target information
        if hasattr(state, "target") and state.target:
            self.current_target = state.target
            self.target_data = self._extract_target_data(state)
        else:
            self.current_target = None
            self.target_data = {}
        
        # Update resource information
        self._update_resources(state)
        
        # Update cooldown information
        self._update_cooldowns()
        
        # Update buffs and debuffs
        self._update_buffs_and_debuffs(state)
    
    def _extract_target_data(self, state: GameState) -> Dict[str, Any]:
        """
        Extract target data from game state
        
        Args:
            state: Current game state
            
        Returns:
            Dict: Target data
        """
        target_data = {}
        
        # Basic target info
        if hasattr(state, "target"):
            target_data["id"] = state.target
        
        if hasattr(state, "target_health"):
            target_data["health_percent"] = state.target_health
        
        if hasattr(state, "target_level"):
            target_data["level"] = state.target_level
        
        if hasattr(state, "target_position"):
            target_data["position"] = state.target_position
        
        if hasattr(state, "target_class"):
            target_data["class"] = state.target_class
        
        if hasattr(state, "target_type"):
            target_data["type"] = state.target_type
        
        # Additional target info from nearby entities
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                if entity.get("id") == state.target:
                    # Merge any additional information from entity data
                    for key, value in entity.items():
                        if key not in target_data:
                            target_data[key] = value
        
        return target_data
    
    def _update_resources(self, state: GameState) -> None:
        """
        Update resource tracking based on game state
        
        Args:
            state: Current game state
        """
        # Default implementation updates basic resources
        if hasattr(state, "player_health"):
            self.current_resources["health"] = state.player_health
        
        if hasattr(state, "player_health_max"):
            self.max_resources["health"] = state.player_health_max
        
        if hasattr(state, "player_mana"):
            self.current_resources["mana"] = state.player_mana
        
        if hasattr(state, "player_mana_max"):
            self.max_resources["mana"] = state.player_mana_max
    
    def _update_cooldowns(self) -> None:
        """Update ability cooldowns based on elapsed time"""
        current_time = time.time()
        
        # Remove expired cooldowns
        expired_abilities = [
            ability for ability, timestamp in self.ability_cooldowns.items()
            if timestamp <= current_time
        ]
        
        for ability in expired_abilities:
            del self.ability_cooldowns[ability]
        
        # Check global cooldown
        if self.global_cooldown <= current_time:
            self.global_cooldown = 0.0
    
    def _update_buffs_and_debuffs(self, state: GameState) -> None:
        """
        Update buff and debuff tracking based on game state
        
        Args:
            state: Current game state
        """
        # Default implementation - derived classes should enhance this
        if hasattr(state, "player_buffs"):
            self.active_buffs = state.player_buffs
        
        if hasattr(state, "target_debuffs"):
            self.target_debuffs = state.target_debuffs
    
    def is_ability_on_cooldown(self, ability_name: str) -> bool:
        """
        Check if an ability is on cooldown
        
        Args:
            ability_name: Name of the ability to check
            
        Returns:
            bool: True if on cooldown, False if available
        """
        return ability_name in self.ability_cooldowns
    
    def is_global_cooldown_active(self) -> bool:
        """
        Check if the global cooldown is active
        
        Returns:
            bool: True if GCD is active, False if available
        """
        return self.global_cooldown > time.time()
    
    def get_resource_percent(self, resource_type: str) -> float:
        """
        Get the percentage of a resource remaining
        
        Args:
            resource_type: Type of resource (health, mana, rage, energy, etc.)
            
        Returns:
            float: Percentage of resource (0-100)
        """
        current = self.current_resources.get(resource_type, 0)
        maximum = self.max_resources.get(resource_type, 100)
        
        if maximum <= 0:
            return 0
        
        return (current / maximum) * 100
    
    def record_ability_use(self, ability_name: str, target_id: Optional[str] = None) -> None:
        """
        Record the use of an ability and update cooldowns
        
        Args:
            ability_name: Name of the ability used
            target_id: ID of the target (if applicable)
        """
        current_time = time.time()
        
        # Record in cast history
        cast_record = {
            "ability": ability_name,
            "target": target_id or self.current_target,
            "timestamp": current_time
        }
        self.cast_history.append(cast_record)
        
        # Trim history to last 20 casts
        if len(self.cast_history) > 20:
            self.cast_history = self.cast_history[-20:]
        
        # Update last cast time
        self.last_cast_time = current_time
        
        # Set global cooldown (default 1.5s)
        self.global_cooldown = current_time + 1.5
        
        # Get ability cooldown from knowledge base
        cooldown_duration = self.knowledge.get_ability_cooldown(ability_name)
        if cooldown_duration > 0:
            self.ability_cooldowns[ability_name] = current_time + cooldown_duration
    
    def calculate_distance_to_target(self) -> float:
        """
        Calculate distance to current target
        
        Returns:
            float: Distance to target or -1 if target position unknown
        """
        if not self.current_target or "position" not in self.target_data:
            return -1
        
        # This would need to be adapted to use the actual player position
        # from the game state when update_state is called
        if not hasattr(self, "player_position") or not self.player_position:
            return -1
        
        # Calculate Euclidean distance
        px, py = self.player_position
        tx, ty = self.target_data["position"]
        
        return ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
    
    def is_buff_active(self, buff_name: str) -> bool:
        """
        Check if a buff is active on the player
        
        Args:
            buff_name: Name of the buff
            
        Returns:
            bool: True if active, False otherwise
        """
        return buff_name in self.active_buffs
    
    def is_debuff_on_target(self, debuff_name: str) -> bool:
        """
        Check if a debuff is active on the current target
        
        Args:
            debuff_name: Name of the debuff
            
        Returns:
            bool: True if active, False otherwise
        """
        return debuff_name in self.target_debuffs
    
    @abstractmethod
    def get_optimal_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal ability rotation based on current state
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        pass
    
    @abstractmethod
    def get_optimal_target(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Get the optimal target for combat
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Dict]: Target information or None
        """
        pass
    
    @abstractmethod
    def get_optimal_position(self, state: GameState) -> Optional[Tuple[float, float]]:
        """
        Get the optimal position for combat
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Tuple]: Position coordinates or None
        """
        pass
    
    @abstractmethod
    def get_resource_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get abilities that should be used to manage resources
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of resource management abilities
        """
        pass
    
    @abstractmethod
    def get_defensive_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get defensive abilities that should be used
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of defensive abilities
        """
        pass
    
    def generate_combat_plan(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Generate a complete combat plan based on the current state
        
        This is the main method that will be called by the CombatManager.
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Combat plan actions
        """
        # Update internal state
        self.update_state(state)
        
        combat_plan = []
        
        try:
            # 1. Check if we need a new target
            if not self.current_target:
                target = self.get_optimal_target(state)
                if target:
                    combat_plan.append({
                        "type": "target",
                        "target": target.get("id"),
                        "description": f"Target {target.get('name', target.get('id'))}"
                    })
            
            # 2. Check if we need to reposition
            if self.current_target:
                position = self.get_optimal_position(state)
                if position:
                    combat_plan.append({
                        "type": "move",
                        "position": position,
                        "description": "Move to optimal combat position"
                    })
            
            # 3. Check for defensive abilities if health is low
            if self.get_resource_percent("health") < 50:
                defensive_abilities = self.get_defensive_abilities(state)
                for ability in defensive_abilities:
                    combat_plan.append({
                        "type": "cast",
                        "spell": ability.get("name"),
                        "target": ability.get("target", "self"),
                        "description": f"Cast defensive {ability.get('name')}"
                    })
            
            # 4. Check for resource management abilities
            resource_abilities = self.get_resource_abilities(state)
            for ability in resource_abilities:
                combat_plan.append({
                    "type": "cast",
                    "spell": ability.get("name"),
                    "target": ability.get("target", "self"),
                    "description": f"Cast resource ability {ability.get('name')}"
                })
            
            # 5. Get the optimal rotation
            rotation = self.get_optimal_rotation(state)
            
            # Add the first few abilities from the rotation
            for ability in rotation[:5]:  # Limit to next 5 abilities
                if not self.is_ability_on_cooldown(ability.get("name")):
                    combat_plan.append({
                        "type": "cast",
                        "spell": ability.get("name"),
                        "target": ability.get("target", self.current_target),
                        "description": f"Cast {ability.get('name')}"
                    })
        
        except Exception as e:
            self.logger.error(f"Error generating combat plan: {e}")
            # Fallback to simple attack if something goes wrong
            if self.current_target:
                combat_plan = [{
                    "type": "cast",
                    "spell": "Attack",
                    "target": self.current_target,
                    "description": "Basic attack (fallback)"
                }]
        
        return combat_plan
    
    @abstractmethod
    def get_supported_talent_builds(self) -> List[Dict[str, Any]]:
        """
        Get the list of supported talent builds for this class
        
        Returns:
            List[Dict]: List of supported talent builds with their rotations
        """
        pass