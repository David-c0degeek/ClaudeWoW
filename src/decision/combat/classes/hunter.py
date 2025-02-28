"""
Hunter Combat Module

This module implements the class-specific combat logic for the Hunter class.
"""

import logging
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.combat.base_combat_module import BaseCombatModule


class HunterCombatModule(BaseCombatModule):
    """
    Hunter-specific combat module implementing the BaseCombatModule interface.
    
    This module handles combat rotations, resource management, and positioning
    specific to the Hunter class in World of Warcraft.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge: GameKnowledge):
        """
        Initialize the Hunter combat module
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        super().__init__(config, knowledge)
        
        # Hunter-specific state tracking
        self.pet_active: bool = False
        self.aspect_active: Optional[str] = None  # hawk, cheetah, viper, etc.
        self.current_stings: Dict[str, float] = {}  # target_id -> sting expiration time
        
        # Add focus resource
        self.current_resources["focus"] = 0
        self.max_resources["focus"] = 100
        
        # Track nearby enemies for AoE decisions
        self.nearby_enemies_count: int = 0
        self.aoe_threshold: int = 3
        
        # Special ability tracking
        self.traps_on_cooldown: Dict[str, float] = {}  # trap name -> cooldown expiry time
        
        self.logger.info("HunterCombatModule initialized")
    
    def update_state(self, state: GameState) -> None:
        """
        Update hunter-specific state
        
        Args:
            state: Current game state
        """
        # Call parent update first
        super().update_state(state)
        
        # Update hunter-specific resources
        self._update_hunter_resources(state)
        
        # Update pet state
        self._update_pet_state(state)
        
        # Update aspect buff
        self._update_aspect_state(state)
        
        # Store player position for distance calculations
        if hasattr(state, "player_position"):
            self.player_position = state.player_position
            
        # Count nearby enemies for AoE decisions
        if hasattr(state, "nearby_entities"):
            self.nearby_enemies_count = sum(
                1 for entity in state.nearby_entities 
                if entity.get("reaction") == "hostile" and
                entity.get("distance", 100) < 10  # Within 10 yards
            )
        
        # Update trap cooldowns
        self._update_trap_cooldowns()
    
    def _update_hunter_resources(self, state: GameState) -> None:
        """
        Update hunter-specific resources like focus
        
        Args:
            state: Current game state
        """
        # Get focus value if available
        if hasattr(state, "player_focus"):
            self.current_resources["focus"] = state.player_focus
        
        if hasattr(state, "player_focus_max"):
            self.max_resources["focus"] = state.player_focus_max
    
    def _update_pet_state(self, state: GameState) -> None:
        """
        Update pet-related information
        
        Args:
            state: Current game state
        """
        # Check if pet is active
        self.pet_active = False
        
        if hasattr(state, "pet_active"):
            self.pet_active = state.pet_active
        elif hasattr(state, "player_buffs") and any("Pet" in buff for buff in state.player_buffs):
            self.pet_active = True
    
    def _update_aspect_state(self, state: GameState) -> None:
        """
        Update which aspect buff is active
        
        Args:
            state: Current game state
        """
        self.aspect_active = None
        
        # Check player buffs for aspects
        if hasattr(state, "player_buffs"):
            for buff in state.player_buffs:
                if "Aspect of" in buff:
                    self.aspect_active = buff.replace("Aspect of ", "").lower()
                    break
    
    def _update_trap_cooldowns(self) -> None:
        """
        Update trap cooldown status
        """
        current_time = time.time()
        expired_traps = [trap for trap, expiry in self.traps_on_cooldown.items() if expiry <= current_time]
        
        for trap in expired_traps:
            del self.traps_on_cooldown[trap]
    
    def get_optimal_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal hunter ability rotation based on specialization and state
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        # Determine the spec from talents or config
        spec = self._determine_specialization(state)
        
        # Generate rotation based on spec
        if spec == "beast_mastery":
            return self._get_beast_mastery_rotation(state)
        elif spec == "marksmanship":
            return self._get_marksmanship_rotation(state)
        elif spec == "survival":
            return self._get_survival_rotation(state)
        else:
            return self._get_leveling_rotation(state)
    
    def _determine_specialization(self, state: GameState) -> str:
        """
        Determine hunter specialization from talents
        
        Args:
            state: Current game state
            
        Returns:
            str: Specialization name (beast_mastery, marksmanship, survival)
        """
        # Check config first
        if "hunter_spec" in self.config:
            return self.config["hunter_spec"]
        
        # Default to beast_mastery if no information is available
        return "beast_mastery"
    
    def _get_beast_mastery_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal Beast Mastery hunter rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if pet needs to be summoned
        if not self.pet_active:
            rotation.append({
                "name": "Call Pet",
                "target": "self",
                "priority": 100,
                "focus_cost": 0,
                "condition": "pet not active"
            })
            
            rotation.append({
                "name": "Revive Pet",
                "target": "self",
                "priority": 99,
                "focus_cost": 0,
                "condition": "pet dead"
            })
        
        # Check for appropriate Aspect
        if self.aspect_active != "hawk":
            rotation.append({
                "name": "Aspect of the Hawk",
                "target": "self",
                "priority": 98,
                "focus_cost": 0,
                "condition": "aspect not active"
            })
        
        # BM cooldowns
        if not self.is_ability_on_cooldown("Bestial Wrath"):
            rotation.append({
                "name": "Bestial Wrath",
                "target": "self",
                "priority": 95,
                "focus_cost": 10,
                "condition": "not on cooldown"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            rotation.append({
                "name": "Multi-Shot",
                "target": self.current_target,
                "priority": 90,
                "focus_cost": 40,
                "condition": "nearby_enemies >= 3"
            })
            
            if not self.is_ability_on_cooldown("Volley"):
                rotation.append({
                    "name": "Volley",
                    "target": "self",
                    "priority": 85,
                    "focus_cost": 20,
                    "condition": "nearby_enemies >= 4"
                })
        
        # Keep Hunter's Mark up
        if not self.is_debuff_on_target("Hunter's Mark"):
            rotation.append({
                "name": "Hunter's Mark",
                "target": self.current_target,
                "priority": 80,
                "focus_cost": 0,
                "condition": "debuff not on target"
            })
        
        # Keep Serpent Sting up
        if not self.is_debuff_on_target("Serpent Sting"):
            rotation.append({
                "name": "Serpent Sting",
                "target": self.current_target,
                "priority": 75,
                "focus_cost": 15,
                "condition": "debuff not on target"
            })
        
        # Rotation core abilities
        if not self.is_ability_on_cooldown("Kill Command"):
            rotation.append({
                "name": "Kill Command",
                "target": self.current_target,
                "priority": 70,
                "focus_cost": 40,
                "condition": "not on cooldown"
            })
        
        if not self.is_ability_on_cooldown("Arcane Shot") and self.current_resources.get("focus", 0) >= 30:
            rotation.append({
                "name": "Arcane Shot",
                "target": self.current_target,
                "priority": 65,
                "focus_cost": 30,
                "condition": "not on cooldown and focus >= 30"
            })
        
        # Focus generator
        rotation.append({
            "name": "Steady Shot",
            "target": self.current_target,
            "priority": 60,
            "focus_cost": -10,  # Generates focus
            "condition": "always"
        })
        
        # Auto shot as fallback
        rotation.append({
            "name": "Auto Shot",
            "target": self.current_target,
            "priority": 0,
            "focus_cost": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_marksmanship_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal Marksmanship hunter rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if pet needs to be summoned
        if not self.pet_active:
            rotation.append({
                "name": "Call Pet",
                "target": "self",
                "priority": 100,
                "focus_cost": 0,
                "condition": "pet not active"
            })
            
            rotation.append({
                "name": "Revive Pet",
                "target": "self",
                "priority": 99,
                "focus_cost": 0,
                "condition": "pet dead"
            })
        
        # Check for appropriate Aspect
        if self.aspect_active != "hawk":
            rotation.append({
                "name": "Aspect of the Hawk",
                "target": "self",
                "priority": 98,
                "focus_cost": 0,
                "condition": "aspect not active"
            })
        
        # MM cooldowns
        if not self.is_ability_on_cooldown("Rapid Fire"):
            rotation.append({
                "name": "Rapid Fire",
                "target": "self",
                "priority": 95,
                "focus_cost": 0,
                "condition": "not on cooldown"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            rotation.append({
                "name": "Multi-Shot",
                "target": self.current_target,
                "priority": 90,
                "focus_cost": 40,
                "condition": "nearby_enemies >= 3"
            })
            
            if not self.is_ability_on_cooldown("Volley"):
                rotation.append({
                    "name": "Volley",
                    "target": "self",
                    "priority": 85,
                    "focus_cost": 20,
                    "condition": "nearby_enemies >= 4"
                })
        
        # Keep Hunter's Mark up
        if not self.is_debuff_on_target("Hunter's Mark"):
            rotation.append({
                "name": "Hunter's Mark",
                "target": self.current_target,
                "priority": 80,
                "focus_cost": 0,
                "condition": "debuff not on target"
            })
        
        # Keep Serpent Sting up
        if not self.is_debuff_on_target("Serpent Sting"):
            rotation.append({
                "name": "Serpent Sting",
                "target": self.current_target,
                "priority": 75,
                "focus_cost": 15,
                "condition": "debuff not on target"
            })
        
        # Rotation core abilities
        if not self.is_ability_on_cooldown("Aimed Shot"):
            rotation.append({
                "name": "Aimed Shot",
                "target": self.current_target,
                "priority": 70,
                "focus_cost": 50,
                "condition": "not on cooldown"
            })
        
        if not self.is_ability_on_cooldown("Chimera Shot"):
            rotation.append({
                "name": "Chimera Shot",
                "target": self.current_target,
                "priority": 65,
                "focus_cost": 45,
                "condition": "not on cooldown"
            })
        
        if not self.is_ability_on_cooldown("Arcane Shot") and self.current_resources.get("focus", 0) >= 30:
            rotation.append({
                "name": "Arcane Shot",
                "target": self.current_target,
                "priority": 60,
                "focus_cost": 30,
                "condition": "not on cooldown and focus >= 30"
            })
        
        # Focus generator
        rotation.append({
            "name": "Steady Shot",
            "target": self.current_target,
            "priority": 55,
            "focus_cost": -10,  # Generates focus
            "condition": "always"
        })
        
        # Auto shot as fallback
        rotation.append({
            "name": "Auto Shot",
            "target": self.current_target,
            "priority": 0,
            "focus_cost": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_survival_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal Survival hunter rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if pet needs to be summoned
        if not self.pet_active:
            rotation.append({
                "name": "Call Pet",
                "target": "self",
                "priority": 100,
                "focus_cost": 0,
                "condition": "pet not active"
            })
            
            rotation.append({
                "name": "Revive Pet",
                "target": "self",
                "priority": 99,
                "focus_cost": 0,
                "condition": "pet dead"
            })
        
        # Check for appropriate Aspect
        if self.aspect_active != "hawk":
            rotation.append({
                "name": "Aspect of the Hawk",
                "target": "self",
                "priority": 98,
                "focus_cost": 0,
                "condition": "aspect not active"
            })
        
        # Set traps if close to target and not on cooldown
        if self.calculate_distance_to_target() < 10:
            available_traps = ["Explosive Trap", "Freezing Trap", "Snake Trap"]
            
            for trap in available_traps:
                if trap not in self.traps_on_cooldown:
                    rotation.append({
                        "name": trap,
                        "target": "ground",
                        "priority": 95,
                        "focus_cost": 0,
                        "condition": f"{trap} not on cooldown"
                    })
                    break
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            if not self.is_ability_on_cooldown("Explosive Shot"):
                rotation.append({
                    "name": "Explosive Shot",
                    "target": self.current_target,
                    "priority": 90,
                    "focus_cost": 25,
                    "condition": "not on cooldown"
                })
            
            rotation.append({
                "name": "Multi-Shot",
                "target": self.current_target,
                "priority": 85,
                "focus_cost": 40,
                "condition": "nearby_enemies >= 3"
            })
            
            if not self.is_ability_on_cooldown("Volley"):
                rotation.append({
                    "name": "Volley",
                    "target": "self",
                    "priority": 80,
                    "focus_cost": 20,
                    "condition": "nearby_enemies >= 4"
                })
        
        # Keep Hunter's Mark up
        if not self.is_debuff_on_target("Hunter's Mark"):
            rotation.append({
                "name": "Hunter's Mark",
                "target": self.current_target,
                "priority": 75,
                "focus_cost": 0,
                "condition": "debuff not on target"
            })
        
        # Keep DoTs up
        if not self.is_debuff_on_target("Serpent Sting"):
            rotation.append({
                "name": "Serpent Sting",
                "target": self.current_target,
                "priority": 70,
                "focus_cost": 15,
                "condition": "debuff not on target"
            })
        
        if not self.is_debuff_on_target("Black Arrow") and not self.is_ability_on_cooldown("Black Arrow"):
            rotation.append({
                "name": "Black Arrow",
                "target": self.current_target,
                "priority": 65,
                "focus_cost": 35,
                "condition": "not on cooldown"
            })
        
        # Rotation core abilities
        if not self.is_ability_on_cooldown("Explosive Shot"):
            rotation.append({
                "name": "Explosive Shot",
                "target": self.current_target,
                "priority": 60,
                "focus_cost": 25,
                "condition": "not on cooldown"
            })
        
        if not self.is_ability_on_cooldown("Kill Shot") and self.target_data.get("health_percent", 100) < 20:
            rotation.append({
                "name": "Kill Shot",
                "target": self.current_target,
                "priority": 55,
                "focus_cost": 15,
                "condition": "not on cooldown and target < 20% health"
            })
        
        if not self.is_ability_on_cooldown("Arcane Shot") and self.current_resources.get("focus", 0) >= 30:
            rotation.append({
                "name": "Arcane Shot",
                "target": self.current_target,
                "priority": 50,
                "focus_cost": 30,
                "condition": "not on cooldown and focus >= 30"
            })
        
        # Focus generator
        rotation.append({
            "name": "Steady Shot",
            "target": self.current_target,
            "priority": 45,
            "focus_cost": -10,  # Generates focus
            "condition": "always"
        })
        
        # Auto shot as fallback
        rotation.append({
            "name": "Auto Shot",
            "target": self.current_target,
            "priority": 0,
            "focus_cost": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_leveling_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get a simple leveling rotation for hunters without many abilities
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if pet needs to be summoned
        if not self.pet_active:
            rotation.append({
                "name": "Call Pet",
                "target": "self",
                "priority": 100,
                "focus_cost": 0,
                "condition": "pet not active"
            })
            
            rotation.append({
                "name": "Revive Pet",
                "target": "self",
                "priority": 99,
                "focus_cost": 0,
                "condition": "pet dead"
            })
        
        # Check for appropriate Aspect
        if not self.aspect_active:
            rotation.append({
                "name": "Aspect of the Hawk",
                "target": "self",
                "priority": 98,
                "focus_cost": 0,
                "condition": "no aspect active"
            })
        
        # Keep Hunter's Mark up
        if not self.is_debuff_on_target("Hunter's Mark"):
            rotation.append({
                "name": "Hunter's Mark",
                "target": self.current_target,
                "priority": 95,
                "focus_cost": 0,
                "condition": "debuff not on target"
            })
        
        # Keep Serpent Sting up
        if not self.is_debuff_on_target("Serpent Sting"):
            rotation.append({
                "name": "Serpent Sting",
                "target": self.current_target,
                "priority": 90,
                "focus_cost": 15,
                "condition": "debuff not on target"
            })
        
        # Use special abilities as they become available
        abilities_to_check = ["Arcane Shot", "Concussive Shot", "Aimed Shot", "Multi-Shot"]
        for ability in abilities_to_check:
            if ability in self.knowledge.get_available_abilities("hunter"):
                rotation.append({
                    "name": ability,
                    "target": self.current_target,
                    "priority": 85,
                    "focus_cost": 30,
                    "condition": "not on cooldown"
                })
        
        # Focus generator
        if "Steady Shot" in self.knowledge.get_available_abilities("hunter"):
            rotation.append({
                "name": "Steady Shot",
                "target": self.current_target,
                "priority": 80,
                "focus_cost": -10,  # Generates focus
                "condition": "always"
            })
        
        # Auto shot as fallback
        rotation.append({
            "name": "Auto Shot",
            "target": self.current_target,
            "priority": 0,
            "focus_cost": 0,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def get_optimal_target(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Get the optimal target for a hunter
        
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
        
        # Sort targets based on specialization and other factors
        # For hunters, distance is less of a concern since they're ranged
        potential_targets.sort(key=lambda e: (
            not any(qt.lower() in e.get("id", "").lower() for qt in self._get_quest_targets(state)),  # Prioritize quest targets
            # For survival hunters, prioritize targets that are a bit closer for trap usage
            e.get("distance", 100) if spec == "survival" else 0,
            # Prioritize targets with Hunter's Mark already applied
            not any("Hunter's Mark" in str(e.get("debuffs", {})))
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
        Get the optimal position for a hunter in combat
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Tuple]: Position coordinates or None
        """
        if not self.current_target or "position" not in self.target_data:
            return None
        
        # Get target position
        tx, ty = self.target_data["position"]
        
        # Calculate current distance to target
        distance = self.calculate_distance_to_target()
        if distance < 0:
            return None  # Can't calculate position without distance
        
        # Determine specialization
        spec = self._determine_specialization(state)
        
        # Optimal ranges vary by spec
        if spec == "survival" and self.calculate_distance_to_target() < 5:
            # Survival should back up to place traps, but not too far
            desired_distance = 15
        elif spec == "beast_mastery":
            # Beast Mastery likes a medium range
            desired_distance = 25
        else:
            # Marksmanship likes to be further away
            desired_distance = 35
        
        # If too close to target, back up
        if distance < desired_distance - 5:
            # Calculate position to move away from target
            if hasattr(self, "player_position") and self.player_position:
                px, py = self.player_position
                
                # Vector from target to player
                dx, dy = px - tx, py - ty
                
                # Normalize the vector
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx, dy = dx/length, dy/length
                    
                    # Position should be at desired distance from target
                    return (tx + dx * desired_distance, ty + dy * desired_distance)
        
        # If already at good range, don't move
        if abs(distance - desired_distance) <= 5:
            return None
        
        # If too far, move closer
        if distance > desired_distance + 5:
            # Calculate position closer to target
            if hasattr(self, "player_position") and self.player_position:
                px, py = self.player_position
                
                # Vector from player to target
                dx, dy = tx - px, ty - py
                
                # Normalize the vector
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx, dy = dx/length, dy/length
                    
                    # Position should be at desired distance from target
                    return (px + dx * (distance - desired_distance), py + dy * (distance - desired_distance))
        
        return None
    
    def get_resource_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get abilities that should be used to manage hunter focus
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of resource management abilities
        """
        abilities = []
        
        # Get focus percentage
        focus_percent = self.get_resource_percent("focus")
        
        # Switch to Viper aspect when low on focus
        if focus_percent < 20 and self.aspect_active != "viper":
            abilities.append({
                "name": "Aspect of the Viper",
                "target": "self",
                "description": "Regenerate focus"
            })
        
        # Switch back to Hawk aspect when focus is restored
        if focus_percent > 80 and self.aspect_active == "viper":
            abilities.append({
                "name": "Aspect of the Hawk",
                "target": "self",
                "description": "Increase damage"
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
        
        # Deterrence when health is critically low and enemies are close
        if health_percent < 20 and self.calculate_distance_to_target() < 10 and not self.is_ability_on_cooldown("Deterrence"):
            defensive_abilities.append({
                "name": "Deterrence",
                "target": "self",
                "description": "Avoid attacks"
            })
        
        # Disengage to create distance
        if self.calculate_distance_to_target() < 10 and not self.is_ability_on_cooldown("Disengage"):
            defensive_abilities.append({
                "name": "Disengage",
                "target": "self",
                "description": "Jump away from target"
            })
        
        # Freezing Trap if enemy is close
        if self.calculate_distance_to_target() < 10 and "Freezing Trap" not in self.traps_on_cooldown:
            defensive_abilities.append({
                "name": "Freezing Trap",
                "target": "ground",
                "description": "Freeze enemy in place"
            })
            self.traps_on_cooldown["Freezing Trap"] = time.time() + 30  # 30-second cooldown
        
        # Feign Death if surrounded by enemies
        if self.nearby_enemies_count >= 3 and not self.is_ability_on_cooldown("Feign Death"):
            defensive_abilities.append({
                "name": "Feign Death",
                "target": "self",
                "description": "Drop threat"
            })
        
        # Mend Pet if pet is low on health
        if hasattr(state, "pet_health") and state.pet_health < 50 and not self.is_ability_on_cooldown("Mend Pet"):
            defensive_abilities.append({
                "name": "Mend Pet",
                "target": "pet",
                "description": "Heal pet"
            })
        
        # Use health potion if health is very low
        if health_percent < 25 and "Health Potion" in self.knowledge.get_available_consumables():
            defensive_abilities.append({
                "name": "Health Potion",
                "target": "self",
                "description": "Use health potion"
            })
        
        return defensive_abilities
    
    def get_supported_talent_builds(self) -> List[Dict[str, Any]]:
        """
        Get the list of supported talent builds for hunters
        
        Returns:
            List[Dict]: List of supported talent builds with their rotations
        """
        return [
            {
                "name": "Beast Mastery",
                "description": "Pet-focused specialization with emphasis on beast damage",
                "key_talents": [
                    "Bestial Wrath",
                    "Ferocious Inspiration",
                    "Improved Aspect of the Hawk",
                    "Unleashed Fury",
                    "Beast Mastery"
                ],
                "rotation_priority": [
                    "Bestial Wrath",
                    "Kill Command",
                    "Serpent Sting",
                    "Arcane Shot",
                    "Steady Shot"
                ]
            },
            {
                "name": "Marksmanship",
                "description": "Precision ranged attacks focusing on aimed shots and critical strikes",
                "key_talents": [
                    "Aimed Shot",
                    "Chimera Shot",
                    "Careful Aim",
                    "Trueshot Aura",
                    "Rapid Killing"
                ],
                "rotation_priority": [
                    "Chimera Shot",
                    "Aimed Shot",
                    "Serpent Sting",
                    "Arcane Shot",
                    "Steady Shot"
                ]
            },
            {
                "name": "Survival",
                "description": "DoT and trap specialist with high burst potential",
                "key_talents": [
                    "Explosive Shot",
                    "Black Arrow",
                    "Lock and Load",
                    "Trap Mastery",
                    "T.N.T."
                ],
                "rotation_priority": [
                    "Explosive Shot",
                    "Black Arrow",
                    "Serpent Sting",
                    "Kill Shot (below 20% health)",
                    "Steady Shot"
                ]
            }
        ]