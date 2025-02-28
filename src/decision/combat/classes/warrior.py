"""
Warrior Combat Module

This module implements the class-specific combat logic for the Warrior class.
"""

import logging
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.combat.base_combat_module import BaseCombatModule


class WarriorCombatModule(BaseCombatModule):
    """
    Warrior-specific combat module implementing the BaseCombatModule interface.
    
    This module handles combat rotations, resource management, and positioning
    specific to the Warrior class in World of Warcraft.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge: GameKnowledge):
        """
        Initialize the Warrior combat module
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        super().__init__(config, knowledge)
        
        # Warrior-specific state tracking
        self.stance: str = "battle"  # battle, defensive, berserker
        self.has_executed_charge: bool = False
        self.current_combo_points: int = 0
        
        # Add rage resource
        self.current_resources["rage"] = 0
        self.max_resources["rage"] = 100
        
        # Track warrior-specific buffs
        self.active_warrior_buffs: Dict[str, Dict[str, Any]] = {}
        
        # Track AoE opportunities
        self.nearby_enemies_count = 0
        self.aoe_threshold = 3  # Number of enemies for AoE abilities
        
        self.logger.info("WarriorCombatModule initialized")
    
    def update_state(self, state: GameState) -> None:
        """
        Update Warrior-specific state
        
        Args:
            state: Current game state
        """
        # Call parent update first
        super().update_state(state)
        
        # Update warrior-specific resources
        self._update_warrior_resources(state)
        
        # Update stance information
        self._update_stance(state)
        
        # Store player position for distance calculations
        if hasattr(state, "player_position"):
            self.player_position = state.player_position
        
        # Reset charge flag if not in combat
        if hasattr(state, "is_in_combat") and not state.is_in_combat:
            self.has_executed_charge = False
            
        # Count nearby enemies for AoE decisions
        if hasattr(state, "nearby_entities"):
            self.nearby_enemies_count = sum(
                1 for entity in state.nearby_entities 
                if entity.get("reaction") == "hostile" and
                entity.get("distance", 100) < 8  # Within 8 yards
            )
    
    def _update_warrior_resources(self, state: GameState) -> None:
        """
        Update warrior-specific resources like rage
        
        Args:
            state: Current game state
        """
        # Get rage value if available
        if hasattr(state, "player_rage"):
            self.current_resources["rage"] = state.player_rage
        
        if hasattr(state, "player_rage_max"):
            self.max_resources["rage"] = state.player_rage_max
    
    def _update_stance(self, state: GameState) -> None:
        """
        Update the current warrior stance
        
        Args:
            state: Current game state
        """
        # Check player buffs for stance
        stance_buffs = {
            "Battle Stance": "battle",
            "Defensive Stance": "defensive",
            "Berserker Stance": "berserker"
        }
        
        # Determine stance from buffs
        if hasattr(state, "player_buffs"):
            for buff_name, stance_name in stance_buffs.items():
                if buff_name in state.player_buffs:
                    self.stance = stance_name
                    break
    
    def get_optimal_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal warrior ability rotation based on specialization and state
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        # Determine the spec from talents or config
        spec = self._determine_specialization(state)
        
        # Generate rotation based on spec
        if spec == "arms":
            return self._get_arms_rotation(state)
        elif spec == "fury":
            return self._get_fury_rotation(state)
        elif spec == "protection":
            return self._get_protection_rotation(state)
        else:
            return self._get_leveling_rotation(state)
    
    def _determine_specialization(self, state: GameState) -> str:
        """
        Determine warrior specialization from talents
        
        Args:
            state: Current game state
            
        Returns:
            str: Specialization name (arms, fury, protection)
        """
        # Check config first
        if "warrior_spec" in self.config:
            return self.config["warrior_spec"]
        
        # Default to arms if no information is available
        return "arms"
    
    def _get_arms_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal arms warrior rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if we should be in the right stance
        if self.stance != "battle":
            rotation.append({
                "name": "Battle Stance",
                "target": "self",
                "priority": 100,
                "condition": "stance != battle"
            })
        
        # Execute phase (target below 20% health)
        if self.target_data and self.target_data.get("health_percent", 100) < 20:
            rotation.append({
                "name": "Execute",
                "target": self.current_target,
                "priority": 90,
                "rage_cost": 15,
                "condition": "target_health < 20%"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            rotation.append({
                "name": "Thunder Clap",
                "target": self.current_target,
                "priority": 85,
                "rage_cost": 20,
                "condition": "nearby_enemies >= 3"
            })
            
            rotation.append({
                "name": "Sweeping Strikes",
                "target": "self",
                "priority": 80,
                "rage_cost": 30,
                "condition": "buff not active"
            })
            
            rotation.append({
                "name": "Cleave",
                "target": self.current_target,
                "priority": 75,
                "rage_cost": 20,
                "condition": "nearby_enemies >= 3"
            })
        
        # Keep Rend up
        if not self.is_debuff_on_target("Rend"):
            rotation.append({
                "name": "Rend",
                "target": self.current_target,
                "priority": 70,
                "rage_cost": 10,
                "condition": "debuff not on target"
            })
        
        # Keep Mortal Strike on cooldown
        if not self.is_ability_on_cooldown("Mortal Strike"):
            rotation.append({
                "name": "Mortal Strike",
                "target": self.current_target,
                "priority": 60,
                "rage_cost": 30,
                "condition": "not on cooldown"
            })
        
        # Overpower is high priority when it procs
        if self.is_buff_active("Overpower"):
            rotation.append({
                "name": "Overpower",
                "target": self.current_target,
                "priority": 50,
                "rage_cost": 5,
                "condition": "proc active"
            })
        
        # Slam with enough rage
        if self.current_resources.get("rage", 0) >= 15:
            rotation.append({
                "name": "Slam",
                "target": self.current_target,
                "priority": 40,
                "rage_cost": 15,
                "condition": "rage >= 15"
            })
        
        # Use Heroic Strike to dump rage
        if self.current_resources.get("rage", 0) >= 30:
            rotation.append({
                "name": "Heroic Strike",
                "target": self.current_target,
                "priority": 30,
                "rage_cost": 15,
                "condition": "rage >= 30"
            })
        
        # Auto attack as fallback
        rotation.append({
            "name": "Attack",
            "target": self.current_target,
            "priority": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_fury_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal fury warrior rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if we should be in the right stance
        if self.stance != "berserker":
            rotation.append({
                "name": "Berserker Stance",
                "target": "self",
                "priority": 100,
                "condition": "stance != berserker"
            })
        
        # Execute phase (target below 20% health)
        if self.target_data and self.target_data.get("health_percent", 100) < 20:
            rotation.append({
                "name": "Execute",
                "target": self.current_target,
                "priority": 90,
                "rage_cost": 15,
                "condition": "target_health < 20%"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            rotation.append({
                "name": "Whirlwind",
                "target": self.current_target,
                "priority": 85,
                "rage_cost": 25,
                "condition": "nearby_enemies >= 3"
            })
            
            rotation.append({
                "name": "Cleave",
                "target": self.current_target,
                "priority": 80,
                "rage_cost": 20,
                "condition": "nearby_enemies >= 3"
            })
        
        # Keep Bloodthirst on cooldown
        if not self.is_ability_on_cooldown("Bloodthirst"):
            rotation.append({
                "name": "Bloodthirst",
                "target": self.current_target,
                "priority": 70,
                "rage_cost": 30,
                "condition": "not on cooldown"
            })
        
        # Whirlwind as part of single-target rotation
        if not self.is_ability_on_cooldown("Whirlwind"):
            rotation.append({
                "name": "Whirlwind",
                "target": self.current_target,
                "priority": 60,
                "rage_cost": 25,
                "condition": "not on cooldown"
            })
        
        # Use Rampage to maintain buff
        if not self.is_buff_active("Rampage") and self.current_resources.get("rage", 0) >= 20:
            rotation.append({
                "name": "Rampage",
                "target": self.current_target,
                "priority": 50,
                "rage_cost": 20,
                "condition": "buff not active"
            })
        
        # Use Heroic Strike to dump rage
        if self.current_resources.get("rage", 0) >= 30:
            rotation.append({
                "name": "Heroic Strike",
                "target": self.current_target,
                "priority": 40,
                "rage_cost": 15,
                "condition": "rage >= 30"
            })
        
        # Auto attack as fallback
        rotation.append({
            "name": "Attack",
            "target": self.current_target,
            "priority": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_protection_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal protection warrior rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if we should be in the right stance
        if self.stance != "defensive":
            rotation.append({
                "name": "Defensive Stance",
                "target": "self",
                "priority": 100,
                "condition": "stance != defensive"
            })
        
        # Keep Shield Block up if not active
        if not self.is_buff_active("Shield Block"):
            rotation.append({
                "name": "Shield Block",
                "target": "self",
                "priority": 90,
                "rage_cost": 10,
                "condition": "buff not active"
            })
        
        # Shield Slam is high priority
        if not self.is_ability_on_cooldown("Shield Slam"):
            rotation.append({
                "name": "Shield Slam",
                "target": self.current_target,
                "priority": 80,
                "rage_cost": 20,
                "condition": "not on cooldown"
            })
        
        # Revenge when it procs
        if self.is_buff_active("Revenge"):
            rotation.append({
                "name": "Revenge",
                "target": self.current_target,
                "priority": 70,
                "rage_cost": 5,
                "condition": "proc active"
            })
        
        # Keep Demoralizing Shout up
        if not self.is_debuff_on_target("Demoralizing Shout"):
            rotation.append({
                "name": "Demoralizing Shout",
                "target": self.current_target,
                "priority": 60,
                "rage_cost": 10,
                "condition": "debuff not on target"
            })
        
        # Thunder Clap for AoE threat/debuff
        if not self.is_ability_on_cooldown("Thunder Clap"):
            rotation.append({
                "name": "Thunder Clap",
                "target": self.current_target,
                "priority": 50,
                "rage_cost": 20,
                "condition": "not on cooldown"
            })
        
        # Heroic Strike to dump rage
        if self.current_resources.get("rage", 0) >= 30:
            rotation.append({
                "name": "Heroic Strike",
                "target": self.current_target,
                "priority": 40,
                "rage_cost": 15,
                "condition": "rage >= 30"
            })
        
        # Auto attack as fallback
        rotation.append({
            "name": "Attack",
            "target": self.current_target,
            "priority": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_leveling_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get a simple leveling rotation for warriors without many abilities
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Use Charge if not in combat
        if not self.has_executed_charge and not state.is_in_combat:
            rotation.append({
                "name": "Charge",
                "target": self.current_target,
                "priority": 100,
                "condition": "not in combat"
            })
        
        # Keep Rend up
        if not self.is_debuff_on_target("Rend"):
            rotation.append({
                "name": "Rend",
                "target": self.current_target,
                "priority": 80,
                "rage_cost": 10,
                "condition": "debuff not on target"
            })
        
        # Thunder Clap for AoE if multiple enemies
        if self.nearby_enemies_count >= 2:
            rotation.append({
                "name": "Thunder Clap",
                "target": self.current_target,
                "priority": 70,
                "rage_cost": 20,
                "condition": "nearby_enemies >= 2"
            })
        
        # Use special abilities as they become available
        abilities_to_check = ["Mortal Strike", "Bloodthirst", "Shield Slam"]
        for ability in abilities_to_check:
            if not self.is_ability_on_cooldown(ability) and ability in self.knowledge.get_available_abilities("warrior"):
                rotation.append({
                    "name": ability,
                    "target": self.current_target,
                    "priority": 60,
                    "rage_cost": 30,
                    "condition": "not on cooldown"
                })
        
        # Heroic Strike to dump rage
        if self.current_resources.get("rage", 0) >= 30:
            rotation.append({
                "name": "Heroic Strike",
                "target": self.current_target,
                "priority": 40,
                "rage_cost": 15,
                "condition": "rage >= 30"
            })
        
        # Auto attack as fallback
        rotation.append({
            "name": "Attack",
            "target": self.current_target,
            "priority": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def get_optimal_target(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Get the optimal target for a warrior
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Dict]: Target information or None
        """
        # Get all possible targets
        potential_targets = []
        
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                # Check if it's a targetable entity
                if entity.get("type") in ["mob", "player"] and entity.get("reaction") == "hostile":
                    potential_targets.append(entity)
        
        if not potential_targets:
            return None
        
        # Determine specialization
        spec = self._determine_specialization(state)
        
        # Sort targets based on specialization
        if spec == "protection":
            # Tank should prioritize multiple targets and targets attacking group members
            potential_targets.sort(key=lambda e: (
                not e.get("attacking_group_member", False),  # Prioritize mobs attacking group members
                e.get("distance", 100),  # Then by distance
                abs(e.get("level", 1) - state.player_level if hasattr(state, "player_level") else 0),  # Then by level difference
                e.get("health_percent", 100)  # Then by health
            ))
        else:
            # DPS specs should prioritize low health targets and quest targets
            potential_targets.sort(key=lambda e: (
                not any(qt.lower() in e.get("id", "").lower() for qt in self._get_quest_targets(state)),  # Prioritize quest targets
                e.get("health_percent", 100),  # Then by health percentage
                e.get("distance", 100),  # Then by distance
                abs(e.get("level", 1) - state.player_level if hasattr(state, "player_level") else 0)  # Then by level difference
            ))
        
        # Return the best target
        if potential_targets:
            return potential_targets[0]
        
        return None
    
    def _get_quest_targets(self, state: GameState) -> List[str]:
        """
        Get list of current quest target names
        
        Args:
            state: Current game state
            
        Returns:
            List[str]: Quest target names
        """
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
        
        return quest_targets
    
    def get_optimal_position(self, state: GameState) -> Optional[Tuple[float, float]]:
        """
        Get the optimal position for a warrior in combat
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Tuple]: Position coordinates or None
        """
        if not self.current_target or "position" not in self.target_data:
            return None
        
        # Get target position
        tx, ty = self.target_data["position"]
        
        # Determine specialization
        spec = self._determine_specialization(state)
        
        # Get current distance to target
        distance = self.calculate_distance_to_target()
        if distance < 0:
            return None  # Can't calculate position without distance
        
        # Calculate optimal position based on spec
        if spec == "protection":
            # Tanks should be right in front of the target
            return (tx, ty)  # Very close
        elif spec in ["arms", "fury"]:
            # Melee DPS should be behind the target if possible
            # Calculate a position behind the target
            if hasattr(self, "player_position") and self.player_position:
                px, py = self.player_position
                
                # Vector from target to player
                dx, dy = px - tx, py - ty
                
                # Normalize the vector
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx, dy = dx/length, dy/length
                    
                    # Position should be 2 yards behind target
                    return (tx - dx * 2, ty - dy * 2)
        
        # Default: If already in melee range, don't move
        if distance <= 5:
            return None
        
        # If not in melee range, move to target
        return (tx, ty)
    
    def get_resource_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get abilities that should be used to manage warrior rage
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of resource management abilities
        """
        abilities = []
        
        # Warriors generate rage in combat, don't typically need resource abilities
        # But we can add Bloodrage
        if self.current_resources.get("rage", 0) < 20 and state.is_in_combat:
            abilities.append({
                "name": "Bloodrage",
                "target": "self",
                "description": "Generate rage"
            })
        
        return abilities
    
    def get_defensive_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get defensive abilities that should be used based on health and situation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of defensive abilities
        """
        defensive_abilities = []
        
        # Get health percentage
        health_percent = self.get_resource_percent("health")
        
        # Last Stand (if health is critically low)
        if health_percent < 20 and not self.is_ability_on_cooldown("Last Stand"):
            defensive_abilities.append({
                "name": "Last Stand",
                "target": "self",
                "description": "Emergency health boost"
            })
        
        # Shield Wall (if health is very low)
        if health_percent < 30 and not self.is_ability_on_cooldown("Shield Wall"):
            defensive_abilities.append({
                "name": "Shield Wall",
                "target": "self",
                "description": "Damage reduction"
            })
        
        # Use health potion if health is low
        if health_percent < 40 and "Health Potion" in self.knowledge.get_available_consumables():
            defensive_abilities.append({
                "name": "Health Potion",
                "target": "self",
                "description": "Use health potion"
            })
        
        # Intercept to stun target (if in berserker stance)
        if self.stance == "berserker" and not self.is_ability_on_cooldown("Intercept"):
            defensive_abilities.append({
                "name": "Intercept",
                "target": self.current_target,
                "description": "Stun target"
            })
        
        # Intimidating Shout to fear (if multiple enemies and health is low)
        if health_percent < 50 and self.nearby_enemies_count >= 2:
            defensive_abilities.append({
                "name": "Intimidating Shout",
                "target": self.current_target,
                "description": "Fear enemies"
            })
        
        return defensive_abilities
    
    def get_supported_talent_builds(self) -> List[Dict[str, Any]]:
        """
        Get the list of supported talent builds for warriors
        
        Returns:
            List[Dict]: List of supported talent builds with their rotations
        """
        return [
            {
                "name": "Arms (2H Weapon)",
                "description": "Two-handed weapon specialist focusing on Mortal Strike",
                "key_talents": [
                    "Tactical Mastery",
                    "Anger Management",
                    "Deep Wounds",
                    "Impale",
                    "Mortal Strike"
                ],
                "rotation_priority": [
                    "Mortal Strike",
                    "Overpower (when it procs)",
                    "Execute (below 20% health)",
                    "Slam",
                    "Heroic Strike (rage dump)"
                ]
            },
            {
                "name": "Fury (Dual Wield)",
                "description": "Dual-wield specialist with high rage generation and sustained damage",
                "key_talents": [
                    "Unbridled Wrath",
                    "Enrage",
                    "Flurry",
                    "Bloodthirst",
                    "Death Wish"
                ],
                "rotation_priority": [
                    "Bloodthirst",
                    "Whirlwind",
                    "Execute (below 20% health)",
                    "Heroic Strike (rage dump)"
                ]
            },
            {
                "name": "Protection (Tank)",
                "description": "Defensive specialist using shield abilities for threat generation",
                "key_talents": [
                    "Shield Specialization",
                    "Defiance",
                    "One-Handed Weapon Specialization",
                    "Shield Slam",
                    "Last Stand"
                ],
                "rotation_priority": [
                    "Shield Slam",
                    "Revenge (when it procs)",
                    "Devastate",
                    "Thunder Clap",
                    "Heroic Strike (rage dump)"
                ]
            }
        ]