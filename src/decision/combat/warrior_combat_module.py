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
    
    def _update_warrior_resources(self, state: GameState) -> None:
        """
        Update warrior-specific resources
        
        Args:
            state: Current game state
        """
        # Update rage if available
        if hasattr(state, "player_rage"):
            self.current_resources["rage"] = state.player_rage
        
        if hasattr(state, "player_rage_max"):
            self.max_resources["rage"] = state.player_rage_max
    
    def _update_stance(self, state: GameState) -> None:
        """
        Update current stance based on game state
        
        Args:
            state: Current game state
        """
        # In a real implementation, we would detect the current stance from buffs
        # or other state information. For now, we'll use a simple heuristic.
        if hasattr(state, "player_buffs"):
            if "Battle Stance" in state.player_buffs:
                self.stance = "battle"
            elif "Defensive Stance" in state.player_buffs:
                self.stance = "defensive"
            elif "Berserker Stance" in state.player_buffs:
                self.stance = "berserker"
    
    def _switch_stance_if_needed(self, target_stance: str, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Create an action to switch stance if needed
        
        Args:
            target_stance: Desired stance
            state: Current game state
            
        Returns:
            Optional[Dict]: Stance switch action or None if already in desired stance
        """
        if self.stance == target_stance:
            return None
        
        stance_ability_map = {
            "battle": "Battle Stance",
            "defensive": "Defensive Stance",
            "berserker": "Berserker Stance"
        }
        
        ability = stance_ability_map.get(target_stance)
        if ability:
            return {
                "name": ability,
                "type": "stance",
                "target": "self",
                "priority": 10.0  # High priority for stance switching
            }
        
        return None
    
    def get_optimal_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal ability rotation for warriors
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        # Get talent build if available
        talent_build = self._detect_talent_build(state)
        
        # Select rotation based on talent build
        if talent_build == "arms":
            return self._get_arms_rotation(state)
        elif talent_build == "fury":
            return self._get_fury_rotation(state)
        elif talent_build == "protection":
            return self._get_protection_rotation(state)
        else:
            # Default to a generic warrior rotation
            return self._get_generic_rotation(state)
    
    def _detect_talent_build(self, state: GameState) -> str:
        """
        Detect the current talent build
        
        Args:
            state: Current game state
            
        Returns:
            str: Detected talent build (arms, fury, protection)
        """
        # In a real implementation, this would parse the talents
        # For now, use a simple heuristic based on stance or config
        
        # Check if the talent build is specified in config
        if "warrior" in self.config and "talent_build" in self.config["warrior"]:
            return self.config["warrior"]["talent_build"]
        
        # Otherwise guess based on stance
        if self.stance == "defensive":
            return "protection"
        elif self.stance == "berserker":
            return "fury"
        else:  # battle stance
            return "arms"
    
    def _get_arms_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal rotation for Arms warriors
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Check if we need to be in Battle Stance
        stance_switch = self._switch_stance_if_needed("battle", state)
        if stance_switch:
            rotation.append(stance_switch)
        
        # Check if we can charge to target
        if not self.has_executed_charge and not self.is_ability_on_cooldown("Charge"):
            distance = self.calculate_distance_to_target()
            if distance > 8 and distance < 25:  # Charge range
                rotation.append({
                    "name": "Charge",
                    "type": "ability",
                    "target": self.current_target,
                    "priority": 9.0,
                    "rage_cost": 0,
                    "rage_gain": 15
                })
                self.has_executed_charge = True
        
        # Check if we need Battle Shout buff
        if not self.is_buff_active("Battle Shout"):
            rotation.append({
                "name": "Battle Shout",
                "type": "buff",
                "target": "self",
                "priority": 8.0,
                "rage_cost": 10,
                "rage_gain": 0
            })
        
        # Main Arms rotation
        
        # Apply Rend if not already applied
        if not self.is_debuff_on_target("Rend"):
            rotation.append({
                "name": "Rend",
                "type": "ability",
                "target": self.current_target,
                "priority": 7.0,
                "rage_cost": 10,
                "rage_gain": 0
            })
        
        # Execute for targets below 20% health
        target_health = self.target_data.get("health_percent", 100)
        if target_health < 20:
            rotation.append({
                "name": "Execute",
                "type": "ability",
                "target": self.current_target,
                "priority": 8.5,
                "rage_cost": 15,
                "rage_gain": 0
            })
        
        # Use Mortal Strike on cooldown
        if not self.is_ability_on_cooldown("Mortal Strike"):
            rotation.append({
                "name": "Mortal Strike",
                "type": "ability",
                "target": self.current_target,
                "priority": 7.5,
                "rage_cost": 30,
                "rage_gain": 0
            })
        
        # Use Overpower if it's available (proc-based)
        # In a real implementation, we would track dodge events
        if not self.is_ability_on_cooldown("Overpower"):
            # This is where we'd check for the "The target dodged your attack" event
            # For now, we'll randomly add it to the rotation with a low chance
            if random.random() < 0.2:
                rotation.append({
                    "name": "Overpower",
                    "type": "ability",
                    "target": self.current_target,
                    "priority": 8.0,
                    "rage_cost": 5,
                    "rage_gain": 0
                })
        
        # Use Heroic Strike when high on rage
        rage_percent = self.get_resource_percent("rage")
        if rage_percent > 60:
            rotation.append({
                "name": "Heroic Strike",
                "type": "ability",
                "target": self.current_target,
                "priority": 6.0,
                "rage_cost": 15,
                "rage_gain": 0
            })
        
        # Use Thunder Clap for multiple targets
        if hasattr(state, "nearby_entities"):
            nearby_enemies = [e for e in state.nearby_entities if e.get("hostile", False)]
            if len(nearby_enemies) >= 3:  # 3+ enemies
                rotation.append({
                    "name": "Thunder Clap",
                    "type": "ability",
                    "target": self.current_target,
                    "priority": 7.0,
                    "rage_cost": 20,
                    "rage_gain": 0
                })
        
        # Basic attack as filler
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "rage_cost": 0,
            "rage_gain": 5  # Auto-attacks generate rage
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def _get_fury_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal rotation for Fury warriors
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Check if we need to be in Berserker Stance
        stance_switch = self._switch_stance_if_needed("berserker", state)
        if stance_switch:
            rotation.append(stance_switch)
        
        # Check if we need Battle Shout buff
        if not self.is_buff_active("Battle Shout"):
            rotation.append({
                "name": "Battle Shout",
                "type": "buff",
                "target": "self",
                "priority": 8.0,
                "rage_cost": 10,
                "rage_gain": 0
            })
        
        # Blood Thirst on cooldown
        if not self.is_ability_on_cooldown("Bloodthirst"):
            rotation.append({
                "name": "Bloodthirst",
                "type": "ability",
                "target": self.current_target,
                "priority": 9.0,
                "rage_cost": 30,
                "rage_gain": 0
            })
        
        # Whirlwind on cooldown
        if not self.is_ability_on_cooldown("Whirlwind"):
            rotation.append({
                "name": "Whirlwind",
                "type": "ability",
                "target": self.current_target,
                "priority": 8.0,
                "rage_cost": 25,
                "rage_gain": 0
            })
        
        # Execute for targets below 20% health
        target_health = self.target_data.get("health_percent", 100)
        if target_health < 20:
            rotation.append({
                "name": "Execute",
                "type": "ability",
                "target": self.current_target,
                "priority": 8.5,
                "rage_cost": 15,
                "rage_gain": 0
            })
        
        # Use Cleave for multiple targets
        if hasattr(state, "nearby_entities"):
            nearby_enemies = [e for e in state.nearby_entities if e.get("hostile", False)]
            if len(nearby_enemies) >= 2:  # 2+ enemies
                rotation.append({
                    "name": "Cleave",
                    "type": "ability",
                    "target": self.current_target,
                    "priority": 7.0,
                    "rage_cost": 20,
                    "rage_gain": 0
                })
        
        # Use Heroic Strike when high on rage
        rage_percent = self.get_resource_percent("rage")
        if rage_percent > 50:  # Fury tends to have more rage to spend
            rotation.append({
                "name": "Heroic Strike",
                "type": "ability",
                "target": self.current_target,
                "priority": 6.0,
                "rage_cost": 15,
                "rage_gain": 0
            })
        
        # Basic attack as filler
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "rage_cost": 0,
            "rage_gain": 5  # Auto-attacks generate rage
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def _get_protection_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal rotation for Protection warriors
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Check if we need to be in Defensive Stance
        stance_switch = self._switch_stance_if_needed("defensive", state)
        if stance_switch:
            rotation.append(stance_switch)
        
        # Check if Shield Block is needed
        if not self.is_buff_active("Shield Block") and not self.is_ability_on_cooldown("Shield Block"):
            rotation.append({
                "name": "Shield Block",
                "type": "buff",
                "target": "self",
                "priority": 9.0,
                "rage_cost": 10,
                "rage_gain": 0
            })
        
        # Apply Sunder Armor to build threat
        if not self.is_debuff_on_target("Sunder Armor") or random.random() < 0.7:  # Maintain stacks
            rotation.append({
                "name": "Sunder Armor",
                "type": "ability",
                "target": self.current_target,
                "priority": 8.0,
                "rage_cost": 15,
                "rage_gain": 0
            })
        
        # Use Revenge when available (proc-based)
        if not self.is_ability_on_cooldown("Revenge"):
            # This is where we'd check for the "Revenge is ready!" proc
            # For now, we'll randomly add it to the rotation with a moderate chance
            if random.random() < 0.5:
                rotation.append({
                    "name": "Revenge",
                    "type": "ability",
                    "target": self.current_target,
                    "priority": 8.5,
                    "rage_cost": 5,
                    "rage_gain": 0
                })
        
        # Use Shield Slam on cooldown
        if not self.is_ability_on_cooldown("Shield Slam"):
            rotation.append({
                "name": "Shield Slam",
                "type": "ability",
                "target": self.current_target,
                "priority": 9.0,
                "rage_cost": 20,
                "rage_gain": 0
            })
        
        # Use Devastate (if available based on level/talents)
        rotation.append({
            "name": "Devastate",
            "type": "ability",
            "target": self.current_target,
            "priority": 7.0,
            "rage_cost": 15,
            "rage_gain": 0
        })
        
        # Use Thunder Clap for multiple targets
        if hasattr(state, "nearby_entities"):
            nearby_enemies = [e for e in state.nearby_entities if e.get("hostile", False)]
            if len(nearby_enemies) >= 3:  # 3+ enemies
                # Switch to battle stance temporarily if needed for Thunder Clap
                if self.stance != "battle" and not self.is_ability_on_cooldown("Battle Stance"):
                    rotation.append({
                        "name": "Battle Stance",
                        "type": "stance",
                        "target": "self",
                        "priority": 7.5,
                        "rage_cost": 0,
                        "rage_gain": 0
                    })
                    
                    rotation.append({
                        "name": "Thunder Clap",
                        "type": "ability",
                        "target": self.current_target,
                        "priority": 7.0,
                        "rage_cost": 20,
                        "rage_gain": 0
                    })
                    
                    # Switch back to defensive stance
                    rotation.append({
                        "name": "Defensive Stance",
                        "type": "stance",
                        "target": "self",
                        "priority": 7.5,
                        "rage_cost": 0,
                        "rage_gain": 0
                    })
        
        # Basic attack as filler
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "rage_cost": 0,
            "rage_gain": 5  # Auto-attacks generate rage
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def _get_generic_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get a generic warrior rotation (for low levels or when talent build is unknown)
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Default to Battle Stance
        stance_switch = self._switch_stance_if_needed("battle", state)
        if stance_switch:
            rotation.append(stance_switch)
        
        # Check if we can charge to target
        if not self.has_executed_charge and not self.is_ability_on_cooldown("Charge"):
            distance = self.calculate_distance_to_target()
            if distance > 8 and distance < 25:  # Charge range
                rotation.append({
                    "name": "Charge",
                    "type": "ability",
                    "target": self.current_target,
                    "priority": 9.0,
                    "rage_cost": 0,
                    "rage_gain": 15
                })
                self.has_executed_charge = True
        
        # Check if we need Battle Shout buff
        if not self.is_buff_active("Battle Shout"):
            rotation.append({
                "name": "Battle Shout",
                "type": "buff",
                "target": "self",
                "priority": 8.0,
                "rage_cost": 10,
                "rage_gain": 0
            })
        
        # Apply Rend if not already applied
        if not self.is_debuff_on_target("Rend"):
            rotation.append({
                "name": "Rend",
                "type": "ability",
                "target": self.current_target,
                "priority": 7.0,
                "rage_cost": 10,
                "rage_gain": 0
            })
        
        # Use Heroic Strike when high on rage
        rage_percent = self.get_resource_percent("rage")
        if rage_percent > 50:
            rotation.append({
                "name": "Heroic Strike",
                "type": "ability",
                "target": self.current_target,
                "priority": 6.0,
                "rage_cost": 15,
                "rage_gain": 0
            })
        
        # Use Thunder Clap for multiple targets
        if hasattr(state, "nearby_entities"):
            nearby_enemies = [e for e in state.nearby_entities if e.get("hostile", False)]
            if len(nearby_enemies) >= 3:  # 3+ enemies
                rotation.append({
                    "name": "Thunder Clap",
                    "type": "ability",
                    "target": self.current_target,
                    "priority": 7.0,
                    "rage_cost": 20,
                    "rage_gain": 0
                })
        
        # Execute for targets below 20% health
        target_health = self.target_data.get("health_percent", 100)
        if target_health < 20:
            rotation.append({
                "name": "Execute",
                "type": "ability",
                "target": self.current_target,
                "priority": 8.5,
                "rage_cost": 15,
                "rage_gain": 0
            })
        
        # Basic attack as filler
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "rage_cost": 0,
            "rage_gain": 5  # Auto-attacks generate rage
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def get_optimal_target(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Get the optimal target for a warrior
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Dict]: Target information or None
        """
        if not hasattr(state, "nearby_entities") or not state.nearby_entities:
            return None
        
        # Look for quest targets first
        quest_targets = []
        if hasattr(state, "active_quests") and state.active_quests:
            for quest in state.active_quests:
                for objective in quest.get("objectives", []):
                    # Extract target name from objective text
                    # This is a simplification - real implementation would be more robust
                    objective_text = objective.get("text", "")
                    if "kill" in objective_text.lower():
                        # Extract target name (simple heuristic)
                        parts = objective_text.split()
                        for i, part in enumerate(parts):
                            if part.lower() == "kill" and i + 1 < len(parts):
                                quest_targets.append(parts[i + 1])
        
        # Filter nearby entities
        suitable_targets = []
        
        for entity in state.nearby_entities:
            entity_name = entity.get("name", "")
            entity_id = entity.get("id", "")
            is_hostile = entity.get("hostile", False)
            
            # Skip non-hostile entities
            if not is_hostile:
                continue
            
            # Calculate distance if positions are available
            distance = -1
            if (hasattr(state, "player_position") and 
                state.player_position and 
                "position" in entity):
                
                px, py = state.player_position
                ex, ey = entity["position"]
                distance = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
            
            # Score this target
            score = 50  # Base score
            
            # Bonus for quest targets
            is_quest_target = any(qt.lower() in entity_name.lower() for qt in quest_targets)
            if is_quest_target:
                score += 50
            
            # Adjust score based on level difference
            if "level" in entity and hasattr(state, "player_level"):
                level_diff = entity["level"] - state.player_level
                if level_diff < -5:
                    score -= 30  # Much lower level
                elif level_diff < -2:
                    score -= 10  # Lower level
                elif level_diff > 3:
                    score -= 40  # Much higher level (avoid)
                elif level_diff > 1:
                    score -= 20  # Higher level
            
            # Adjust score based on health
            if "health_percent" in entity:
                health_percent = entity["health_percent"]
                if health_percent < 30:
                    score += 20  # Low health targets are easier
            
            # Adjust score based on distance (prefer closer targets)
            if distance > 0:
                if distance < 5:
                    score += 15  # Very close
                elif distance < 15:
                    score += 5  # Close enough
                elif distance > 30:
                    score -= 30  # Too far
            
            suitable_targets.append({
                "id": entity_id,
                "name": entity_name,
                "score": score,
                "distance": distance,
                "level": entity.get("level", 0),
                "health_percent": entity.get("health_percent", 100)
            })
        
        # Sort by score (highest first)
        suitable_targets.sort(key=lambda x: x["score"], reverse=True)
        
        return suitable_targets[0] if suitable_targets else None
    
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
        target_x, target_y = self.target_data["position"]
        
        # Warriors need to be in melee range
        melee_range = 3.0
        
        # Get current distance to target
        current_distance = self.calculate_distance_to_target()
        
        # If we're close enough, no need to move
        if 0 < current_distance <= melee_range:
            return None
        
        # Generate position in melee range
        # Add a slight random offset to avoid stacking exactly on the target
        angle = random.uniform(0, 2 * math.pi)
        offset_x = melee_range * math.cos(angle)
        offset_y = melee_range * math.sin(angle)
        
        return (target_x + offset_x, target_y + offset_y)
    
    def get_resource_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get abilities for resource management (rage generation)
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of resource management abilities
        """
        resource_abilities = []
        
        # Check current rage
        rage_percent = self.get_resource_percent("rage")
        
        # Warriors don't have many abilities to generate rage outside of combat
        # Mostly they rely on auto-attacks and taking damage
        
        # Bloodrage can be used to generate rage
        if rage_percent < 30 and not self.is_ability_on_cooldown("Bloodrage"):
            resource_abilities.append({
                "name": "Bloodrage",
                "type": "ability",
                "target": "self",
                "priority": 5.0,
                "description": "Generate rage"
            })
        
        return resource_abilities
    
    def get_defensive_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get defensive abilities for warriors
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of defensive abilities
        """
        defensive_abilities = []
        
        # Check health percentage
        health_percent = self.get_resource_percent("health")
        
        # Last Stand (if available)
        if health_percent < 30 and not self.is_ability_on_cooldown("Last Stand"):
            defensive_abilities.append({
                "name": "Last Stand",
                "type": "defensive",
                "target": "self",
                "priority": 9.0,
                "description": "Increase max health"
            })
        
        # Shield Wall (if in defensive stance)
        if health_percent < 20 and self.stance == "defensive" and not self.is_ability_on_cooldown("Shield Wall"):
            defensive_abilities.append({
                "name": "Shield Wall",
                "type": "defensive",
                "target": "self",
                "priority": 10.0,
                "description": "Reduce damage taken"
            })
        
        # Intimidating Shout to CC enemies when low on health
        if health_percent < 25 and not self.is_ability_on_cooldown("Intimidating Shout"):
            defensive_abilities.append({
                "name": "Intimidating Shout",
                "type": "defensive",
                "target": self.current_target,
                "priority": 8.0,
                "description": "Fear nearby enemies"
            })
        
        return defensive_abilities
    
    def get_supported_talent_builds(self) -> List[Dict[str, Any]]:
        """
        Get supported talent builds for warriors
        
        Returns:
            List[Dict]: List of supported talent builds with their rotations
        """
        return [
            {
                "name": "Arms",
                "description": "Two-handed weapon specialist with high burst damage",
                "key_abilities": ["Mortal Strike", "Overpower", "Execute"],
                "preferred_stance": "battle",
                "weapons": "Two-handed weapons"
            },
            {
                "name": "Fury",
                "description": "Dual-wield specialist with high sustained damage",
                "key_abilities": ["Bloodthirst", "Whirlwind", "Execute"],
                "preferred_stance": "berserker",
                "weapons": "One-handed weapons (dual-wielding)"
            },
            {
                "name": "Protection",
                "description": "Tank specialization with high threat and survivability",
                "key_abilities": ["Shield Slam", "Revenge", "Sunder Armor"],
                "preferred_stance": "defensive",
                "weapons": "One-handed weapon and shield"
            }
        ]