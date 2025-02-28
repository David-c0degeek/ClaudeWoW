"""
Mage Combat Module

This module implements the class-specific combat logic for the Mage class.
"""

import logging
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.combat.base_combat_module import BaseCombatModule


class MageCombatModule(BaseCombatModule):
    """
    Mage-specific combat module implementing the BaseCombatModule interface.
    
    This module handles combat rotations, resource management, and positioning
    specific to the Mage class in World of Warcraft.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge: GameKnowledge):
        """
        Initialize the Mage combat module
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        super().__init__(config, knowledge)
        
        # Mage-specific state tracking
        self.active_frost_debuffs: List[str] = []
        self.active_fire_debuffs: List[str] = []
        self.has_proc_brain_freeze: bool = False
        self.has_proc_hot_streak: bool = False
        self.has_proc_fingers_of_frost: bool = False
        self.polymorph_targets: Dict[str, float] = {}  # Target ID -> expiry time
        self.clearcasting_active: bool = False
        
        # Add mana resource
        self.current_resources["mana"] = 0
        self.max_resources["mana"] = 100
        
        # Track nearby enemies for AoE decisions
        self.nearby_enemies_count: int = 0
        self.aoe_threshold: int = 3
        
        # Mage-specific buffs
        self.active_mage_buffs: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("MageCombatModule initialized")
    
    def update_state(self, state: GameState) -> None:
        """
        Update mage-specific state
        
        Args:
            state: Current game state
        """
        # Call parent update first
        super().update_state(state)
        
        # Update mage-specific resources
        self._update_mage_resources(state)
        
        # Update proc states
        self._update_proc_states(state)
        
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
            
        # Update polymorph target timers
        current_time = time.time()
        expired_poly = [target_id for target_id, expiry in self.polymorph_targets.items() 
                        if expiry <= current_time]
        for target_id in expired_poly:
            del self.polymorph_targets[target_id]
    
    def _update_mage_resources(self, state: GameState) -> None:
        """
        Update mage-specific resources like mana
        
        Args:
            state: Current game state
        """
        # Get mana value if available
        if hasattr(state, "player_mana"):
            self.current_resources["mana"] = state.player_mana
        
        if hasattr(state, "player_mana_max"):
            self.max_resources["mana"] = state.player_mana_max
    
    def _update_proc_states(self, state: GameState) -> None:
        """
        Update mage proc states like Hot Streak, Brain Freeze, etc.
        
        Args:
            state: Current game state
        """
        # Check player buffs for procs
        proc_buffs = {
            "Brain Freeze": "has_proc_brain_freeze",
            "Hot Streak": "has_proc_hot_streak", 
            "Fingers of Frost": "has_proc_fingers_of_frost",
            "Clearcasting": "clearcasting_active"
        }
        
        # Reset all proc states first
        for _, attr_name in proc_buffs.items():
            setattr(self, attr_name, False)
        
        # Set proc states based on active buffs
        if hasattr(state, "player_buffs"):
            for buff_name, attr_name in proc_buffs.items():
                if buff_name in state.player_buffs:
                    setattr(self, attr_name, True)
    
    def get_optimal_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal mage ability rotation based on specialization and state
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        # Determine the spec from talents or config
        spec = self._determine_specialization(state)
        
        # Generate rotation based on spec
        if spec == "frost":
            return self._get_frost_rotation(state)
        elif spec == "fire":
            return self._get_fire_rotation(state)
        elif spec == "arcane":
            return self._get_arcane_rotation(state)
        else:
            return self._get_leveling_rotation(state)
    
    def _determine_specialization(self, state: GameState) -> str:
        """
        Determine mage specialization from talents
        
        Args:
            state: Current game state
            
        Returns:
            str: Specialization name (frost, fire, arcane)
        """
        # Check config first
        if "mage_spec" in self.config:
            return self.config["mage_spec"]
        
        # Default to frost if no information is available
        return "frost"
    
    def _get_frost_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal frost mage rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Check if Water Elemental is summoned
        if not self.is_buff_active("Water Elemental") and not self.is_ability_on_cooldown("Summon Water Elemental"):
            rotation.append({
                "name": "Summon Water Elemental",
                "target": "self",
                "priority": 100,
                "mana_cost": 16,
                "condition": "pet not active"
            })
        
        # Check for Ice Barrier if not active
        if not self.is_buff_active("Ice Barrier") and not self.is_ability_on_cooldown("Ice Barrier"):
            rotation.append({
                "name": "Ice Barrier",
                "target": "self",
                "priority": 95,
                "mana_cost": 12,
                "condition": "buff not active"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            # Frozen Orb for AoE and procs
            if not self.is_ability_on_cooldown("Frozen Orb"):
                rotation.append({
                    "name": "Frozen Orb",
                    "target": self.current_target,
                    "priority": 90,
                    "mana_cost": 10,
                    "condition": "not on cooldown"
                })
            
            # Blizzard for sustained AoE
            rotation.append({
                "name": "Blizzard",
                "target": self.current_target,
                "priority": 85,
                "mana_cost": 12,
                "condition": "nearby_enemies >= 3"
            })
            
            # Cone of Cold if enemies are clumped
            rotation.append({
                "name": "Cone of Cold",
                "target": self.current_target,
                "priority": 80,
                "mana_cost": 10,
                "condition": "nearby_enemies >= 3"
            })
        
        # Use Brain Freeze proc
        if self.has_proc_brain_freeze:
            rotation.append({
                "name": "Glacial Spike",
                "target": self.current_target,
                "priority": 75,
                "mana_cost": 0,  # Free due to proc
                "condition": "brain_freeze active"
            })
        
        # Use Fingers of Frost proc
        if self.has_proc_fingers_of_frost:
            rotation.append({
                "name": "Ice Lance",
                "target": self.current_target,
                "priority": 70,
                "mana_cost": 0,  # Free due to proc
                "condition": "fingers_of_frost active"
            })
        
        # Keep Frostbite up
        if not self.is_debuff_on_target("Frostbite"):
            rotation.append({
                "name": "Frostbolt",
                "target": self.current_target,
                "priority": 65,
                "mana_cost": 4,
                "condition": "debuff not on target"
            })
        
        # Use Icy Veins on cooldown for burst
        if not self.is_ability_on_cooldown("Icy Veins") and not self.is_buff_active("Icy Veins"):
            rotation.append({
                "name": "Icy Veins",
                "target": "self",
                "priority": 60,
                "mana_cost": 3,
                "condition": "not on cooldown"
            })
        
        # Use Frozen Orb on cooldown
        if not self.is_ability_on_cooldown("Frozen Orb"):
            rotation.append({
                "name": "Frozen Orb",
                "target": self.current_target,
                "priority": 55,
                "mana_cost": 10,
                "condition": "not on cooldown"
            })
        
        # Frostbolt as main filler
        rotation.append({
            "name": "Frostbolt",
            "target": self.current_target,
            "priority": 50,
            "mana_cost": 4,
            "condition": "always"
        })
        
        # Ice Lance as filler when moving
        rotation.append({
            "name": "Ice Lance",
            "target": self.current_target,
            "priority": 45,
            "mana_cost": 2,
            "condition": "while moving"
        })
        
        # Fire Blast as emergency damage
        if not self.is_ability_on_cooldown("Fire Blast"):
            rotation.append({
                "name": "Fire Blast",
                "target": self.current_target,
                "priority": 40,
                "mana_cost": 5,
                "condition": "not on cooldown"
            })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_fire_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal fire mage rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Combat opener
        if not state.is_in_combat:
            # Start with Pyroblast
            rotation.append({
                "name": "Pyroblast",
                "target": self.current_target,
                "priority": 100,
                "mana_cost": 10,
                "condition": "not in combat"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            # Flamestrike for AoE
            rotation.append({
                "name": "Flamestrike",
                "target": self.current_target,
                "priority": 95,
                "mana_cost": 12,
                "condition": "nearby_enemies >= 3"
            })
            
            # Dragon's Breath if enemies are clumped
            if not self.is_ability_on_cooldown("Dragon's Breath"):
                rotation.append({
                    "name": "Dragon's Breath",
                    "target": self.current_target,
                    "priority": 90,
                    "mana_cost": 7,
                    "condition": "nearby_enemies >= 3"
                })
        
        # Use Combustion cooldown for burst
        if not self.is_ability_on_cooldown("Combustion") and not self.is_buff_active("Combustion"):
            rotation.append({
                "name": "Combustion",
                "target": "self",
                "priority": 85,
                "mana_cost": 10,
                "condition": "not on cooldown"
            })
        
        # Use Hot Streak proc
        if self.has_proc_hot_streak:
            rotation.append({
                "name": "Pyroblast",
                "target": self.current_target,
                "priority": 80,
                "mana_cost": 0,  # Free due to proc
                "condition": "hot_streak active"
            })
        
        # Keep Living Bomb up on multiple targets
        if not self.is_debuff_on_target("Living Bomb") and self.nearby_enemies_count > 1:
            rotation.append({
                "name": "Living Bomb",
                "target": self.current_target,
                "priority": 75,
                "mana_cost": 7,
                "condition": "debuff not on target"
            })
        
        # Fire Blast to get Hot Streak proc
        if not self.is_ability_on_cooldown("Fire Blast"):
            rotation.append({
                "name": "Fire Blast",
                "target": self.current_target,
                "priority": 70,
                "mana_cost": 5,
                "condition": "not on cooldown"
            })
        
        # Phoenix Flames to help get Hot Streak
        if not self.is_ability_on_cooldown("Phoenix Flames"):
            rotation.append({
                "name": "Phoenix Flames",
                "target": self.current_target,
                "priority": 65,
                "mana_cost": 8,
                "condition": "not on cooldown"
            })
        
        # Scorch if moving
        rotation.append({
            "name": "Scorch",
            "target": self.current_target,
            "priority": 60,
            "mana_cost": 3,
            "condition": "while moving"
        })
        
        # Fireball as main filler
        rotation.append({
            "name": "Fireball",
            "target": self.current_target,
            "priority": 55,
            "mana_cost": 5,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_arcane_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal arcane mage rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Mana management threshold
        mana_percent = self.get_resource_percent("mana")
        low_mana = mana_percent < 30
        
        # Check for Arcane Power
        if not self.is_ability_on_cooldown("Arcane Power") and not self.is_buff_active("Arcane Power") and not low_mana:
            rotation.append({
                "name": "Arcane Power",
                "target": "self",
                "priority": 100,
                "mana_cost": 8,
                "condition": "not on cooldown"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            # Arcane Explosion for AoE
            rotation.append({
                "name": "Arcane Explosion",
                "target": self.current_target,
                "priority": 95,
                "mana_cost": 10,
                "condition": "nearby_enemies >= 3"
            })
        
        # Use Evocation when low on mana
        if low_mana and not self.is_ability_on_cooldown("Evocation"):
            rotation.append({
                "name": "Evocation",
                "target": "self",
                "priority": 90,
                "mana_cost": 0,
                "condition": "mana < 30%"
            })
        
        # Use Arcane Barrage to clear stacks when low on mana
        if low_mana:
            rotation.append({
                "name": "Arcane Barrage",
                "target": self.current_target,
                "priority": 85,
                "mana_cost": 3,
                "condition": "mana < 30%"
            })
        
        # Use Arcane Missiles when Clearcasting procs
        if self.clearcasting_active:
            rotation.append({
                "name": "Arcane Missiles",
                "target": self.current_target,
                "priority": 80,
                "mana_cost": 0,  # Free due to clearcasting
                "condition": "clearcasting active"
            })
        
        # Use Presence of Mind with Arcane Blast for burst
        if not self.is_ability_on_cooldown("Presence of Mind") and not low_mana:
            rotation.append({
                "name": "Presence of Mind",
                "target": "self",
                "priority": 75,
                "mana_cost": 0,
                "condition": "not on cooldown"
            })
            rotation.append({
                "name": "Arcane Blast",
                "target": self.current_target,
                "priority": 74,
                "mana_cost": 7,
                "condition": "presence_of_mind active"
            })
        
        # Arcane Blast as main spender (when not low on mana)
        if not low_mana:
            rotation.append({
                "name": "Arcane Blast",
                "target": self.current_target,
                "priority": 70,
                "mana_cost": 7,
                "condition": "mana >= 30%"
            })
        
        # Arcane Barrage to clear Arcane Charges occasionally
        rotation.append({
            "name": "Arcane Barrage",
            "target": self.current_target,
            "priority": 65,
            "mana_cost": 3,
            "condition": "arcane_charges >= 4"
        })
        
        # Arcane Missiles as filler
        rotation.append({
            "name": "Arcane Missiles",
            "target": self.current_target,
            "priority": 60,
            "mana_cost": 5,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_leveling_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get a simple leveling rotation for mages without many abilities
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Frost Nova if enemies get close
        if self.calculate_distance_to_target() < 5 and not self.is_ability_on_cooldown("Frost Nova"):
            rotation.append({
                "name": "Frost Nova",
                "target": "self",
                "priority": 100,
                "mana_cost": 8,
                "condition": "enemy too close"
            })
        
        # Fire Blast as instant damage
        if not self.is_ability_on_cooldown("Fire Blast"):
            rotation.append({
                "name": "Fire Blast",
                "target": self.current_target,
                "priority": 90,
                "mana_cost": 5,
                "condition": "not on cooldown"
            })
        
        # AoE with Arcane Explosion if multiple enemies
        if self.nearby_enemies_count >= 3:
            rotation.append({
                "name": "Arcane Explosion",
                "target": "self",
                "priority": 80,
                "mana_cost": 10,
                "condition": "nearby_enemies >= 3"
            })
        
        # Use primary nuke based on level
        abilities_to_check = ["Frostbolt", "Fireball", "Arcane Missiles"]
        for ability in abilities_to_check:
            if ability in self.knowledge.get_available_abilities("mage"):
                rotation.append({
                    "name": ability,
                    "target": self.current_target,
                    "priority": 70,
                    "mana_cost": 5,
                    "condition": "always"
                })
                break
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def get_optimal_target(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Get the optimal target for a mage
        
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
                    # Skip polymorphed targets
                    if entity.get("id") in self.polymorph_targets:
                        continue
                    potential_targets.append(entity)
        
        if not potential_targets:
            return None
        
        # Determine specialization
        spec = self._determine_specialization(state)
        
        # Sort targets based on specialization
        if spec == "frost":
            # Frost should prioritize targets that are already slowed
            potential_targets.sort(key=lambda e: (
                not any(debuff in str(e.get("debuffs", {})) for debuff in ["Frostbolt", "Cone of Cold"]),
                e.get("distance", 100),  # Then by distance
                abs(e.get("level", 1) - state.player_level if hasattr(state, "player_level") else 0)  # Then by level difference
            ))
        elif spec == "fire":
            # Fire should prioritize targets that are already burning
            potential_targets.sort(key=lambda e: (
                not any(debuff in str(e.get("debuffs", {})) for debuff in ["Fireball", "Living Bomb"]),
                e.get("distance", 100),  # Then by distance
                abs(e.get("level", 1) - state.player_level if hasattr(state, "player_level") else 0)  # Then by level difference
            ))
        else:
            # Default sorting
            potential_targets.sort(key=lambda e: (
                not any(qt.lower() in e.get("id", "").lower() for qt in self._get_quest_targets(state)),  # Prioritize quest targets
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
        Get the optimal position for a mage in combat
        
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
        
        # If we're too close to the target, back up
        if distance < 15:
            # Calculate position to move away from target
            if hasattr(self, "player_position") and self.player_position:
                px, py = self.player_position
                
                # Vector from target to player
                dx, dy = px - tx, py - ty
                
                # Normalize the vector
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx, dy = dx/length, dy/length
                    
                    # Position should be about 30 yards away from target
                    desired_distance = 30
                    return (tx + dx * desired_distance, ty + dy * desired_distance)
        
        # If already at good range (15-40 yards), don't move
        if 15 <= distance <= 40:
            return None
        
        # If too far, move closer but maintain safe distance
        # Calculate position 25 yards away
        if hasattr(self, "player_position") and self.player_position:
            px, py = self.player_position
            
            # Vector from player to target
            dx, dy = tx - px, ty - py
            
            # Normalize the vector
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx, dy = dx/length, dy/length
                
                # Position should be about 25 yards away from target
                desired_distance = 25
                return (px + dx * (distance - desired_distance), py + dy * (distance - desired_distance))
        
        return None
    
    def get_resource_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get abilities that should be used to manage mage mana
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of resource management abilities
        """
        abilities = []
        
        # Get mana percentage
        mana_percent = self.get_resource_percent("mana")
        
        # Use Evocation when mana is low
        if mana_percent < 20 and not self.is_ability_on_cooldown("Evocation"):
            abilities.append({
                "name": "Evocation",
                "target": "self",
                "description": "Regenerate mana"
            })
        
        # Use Mana Gem when mana is moderately low
        if 20 <= mana_percent < 50 and "Mana Gem" in self.knowledge.get_available_consumables():
            abilities.append({
                "name": "Mana Gem",
                "target": "self",
                "description": "Use mana gem"
            })
        
        # Use Arcane Intellect if not already buffed
        if not self.is_buff_active("Arcane Intellect"):
            abilities.append({
                "name": "Arcane Intellect",
                "target": "self",
                "description": "Buff intellect"
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
        
        # Ice Block (emergency survival)
        if health_percent < 15 and not self.is_ability_on_cooldown("Ice Block"):
            defensive_abilities.append({
                "name": "Ice Block",
                "target": "self",
                "description": "Emergency immunity"
            })
        
        # Frost Nova (if enemies are close)
        if self.calculate_distance_to_target() < 8 and not self.is_ability_on_cooldown("Frost Nova"):
            defensive_abilities.append({
                "name": "Frost Nova",
                "target": "self",
                "description": "Freeze nearby enemies"
            })
        
        # Ice Barrier if not active
        if not self.is_buff_active("Ice Barrier") and not self.is_ability_on_cooldown("Ice Barrier"):
            defensive_abilities.append({
                "name": "Ice Barrier",
                "target": "self",
                "description": "Shield against damage"
            })
        
        # Blink to create distance
        if self.calculate_distance_to_target() < 10 and not self.is_ability_on_cooldown("Blink"):
            defensive_abilities.append({
                "name": "Blink",
                "target": "self",
                "description": "Teleport away from danger"
            })
        
        # Polymorph adds
        if self.nearby_enemies_count >= 2 and not self.is_ability_on_cooldown("Polymorph"):
            # Find a secondary target that isn't the main target
            secondary_target = None
            if hasattr(state, "nearby_entities") and state.nearby_entities:
                for entity in state.nearby_entities:
                    if (entity.get("type") in ["mob"] and 
                        entity.get("reaction") == "hostile" and
                        entity.get("id") != self.current_target and
                        entity.get("id") not in self.polymorph_targets):
                        secondary_target = entity
                        break
            
            if secondary_target:
                defensive_abilities.append({
                    "name": "Polymorph",
                    "target": secondary_target.get("id"),
                    "description": "Crowd control add"
                })
                # Set polymorph expiry (30 seconds)
                self.polymorph_targets[secondary_target.get("id")] = time.time() + 30
        
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
        Get the list of supported talent builds for mages
        
        Returns:
            List[Dict]: List of supported talent builds with their rotations
        """
        return [
            {
                "name": "Frost (Control)",
                "description": "Frost mage focusing on control and consistent damage",
                "key_talents": [
                    "Icy Veins",
                    "Ice Barrier",
                    "Frost Channeling",
                    "Shatter",
                    "Ice Lance"
                ],
                "rotation_priority": [
                    "Frozen Orb",
                    "Frostbolt",
                    "Ice Lance (with Fingers of Frost)",
                    "Glacial Spike (with Brain Freeze)",
                    "Blizzard (AoE)"
                ]
            },
            {
                "name": "Fire (Burst Damage)",
                "description": "Fire mage focusing on critical strikes and burst damage",
                "key_talents": [
                    "Pyroblast",
                    "Hot Streak",
                    "Critical Mass",
                    "Combustion",
                    "Living Bomb"
                ],
                "rotation_priority": [
                    "Pyroblast (with Hot Streak)",
                    "Fire Blast (to generate Hot Streak)",
                    "Phoenix Flames",
                    "Fireball",
                    "Flamestrike (AoE)"
                ]
            },
            {
                "name": "Arcane (Mana Management)",
                "description": "Arcane mage focusing on mana management and burst windows",
                "key_talents": [
                    "Arcane Power",
                    "Presence of Mind",
                    "Arcane Concentration",
                    "Arcane Barrage",
                    "Evocation"
                ],
                "rotation_priority": [
                    "Arcane Blast (during burst)",
                    "Arcane Missiles (with Clearcasting)",
                    "Arcane Barrage (to clear stacks)",
                    "Evocation (mana recovery)",
                    "Arcane Explosion (AoE)"
                ]
            }
        ]