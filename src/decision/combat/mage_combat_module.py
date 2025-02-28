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
        
        # Add mana resource
        self.current_resources["mana"] = 0
        self.max_resources["mana"] = 100
        
        # Mage-specific buffs
        self.active_mage_buffs: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("MageCombatModule initialized")
    
    def update_state(self, state: GameState) -> None:
        """
        Update Mage-specific state
        
        Args:
            state: Current game state
        """
        # Call parent update first
        super().update_state(state)
        
        # Update mage-specific resources
        self._update_mage_resources(state)
        
        # Update procs and debuffs
        self._update_procs_and_debuffs(state)
        
        # Store player position for distance calculations
        if hasattr(state, "player_position"):
            self.player_position = state.player_position
        
        # Update polymorph targets (remove expired)
        current_time = time.time()
        expired_targets = [
            target_id for target_id, expiry_time in self.polymorph_targets.items()
            if expiry_time <= current_time
        ]
        for target_id in expired_targets:
            del self.polymorph_targets[target_id]
    
    def _update_mage_resources(self, state: GameState) -> None:
        """
        Update mage-specific resources
        
        Args:
            state: Current game state
        """
        # Update mana if available
        if hasattr(state, "player_mana"):
            self.current_resources["mana"] = state.player_mana
        
        if hasattr(state, "player_mana_max"):
            self.max_resources["mana"] = state.player_mana_max
    
    def _update_procs_and_debuffs(self, state: GameState) -> None:
        """
        Update mage proc states and active debuffs
        
        Args:
            state: Current game state
        """
        # In a real implementation, we would check for specific buffs and debuffs
        # For now, we'll use a simple heuristic
        
        # Check for procs in player buffs
        if hasattr(state, "player_buffs") and state.player_buffs:
            # Update procs
            self.has_proc_brain_freeze = "Brain Freeze" in state.player_buffs
            self.has_proc_hot_streak = "Hot Streak" in state.player_buffs
            self.has_proc_fingers_of_frost = "Fingers of Frost" in state.player_buffs
        
        # Update target debuffs
        if hasattr(state, "target_debuffs") and state.target_debuffs:
            # Check frost debuffs
            self.active_frost_debuffs = [
                debuff for debuff in ["Frostbolt", "Frost Nova", "Winter's Chill"]
                if debuff in state.target_debuffs
            ]
            
            # Check fire debuffs
            self.active_fire_debuffs = [
                debuff for debuff in ["Pyroblast", "Living Bomb", "Ignite"]
                if debuff in state.target_debuffs
            ]
    
    def get_optimal_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal ability rotation for mages
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        # Get talent build if available
        talent_build = self._detect_talent_build(state)
        
        # Select rotation based on talent build
        if talent_build == "frost":
            return self._get_frost_rotation(state)
        elif talent_build == "fire":
            return self._get_fire_rotation(state)
        elif talent_build == "arcane":
            return self._get_arcane_rotation(state)
        else:
            # Default to a generic mage rotation
            return self._get_generic_rotation(state)
    
    def _detect_talent_build(self, state: GameState) -> str:
        """
        Detect the current talent build
        
        Args:
            state: Current game state
            
        Returns:
            str: Detected talent build (frost, fire, arcane)
        """
        # In a real implementation, this would parse the talents
        # For now, use a simple heuristic based on config or buffs
        
        # Check if the talent build is specified in config
        if "mage" in self.config and "talent_build" in self.config["mage"]:
            return self.config["mage"]["talent_build"]
        
        # Otherwise guess based on active buffs and debuffs
        if len(self.active_frost_debuffs) > len(self.active_fire_debuffs):
            return "frost"
        elif len(self.active_fire_debuffs) > len(self.active_frost_debuffs):
            return "fire"
        elif hasattr(state, "player_buffs") and "Arcane Power" in state.player_buffs:
            return "arcane"
        
        # Default to frost as a common leveling spec
        return "frost"
    
    def _get_frost_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal rotation for Frost mages
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Check if we need to apply buffs
        if not self.is_buff_active("Arcane Intellect"):
            rotation.append({
                "name": "Arcane Intellect",
                "type": "buff",
                "target": "self",
                "priority": 9.0,
                "mana_cost": 500
            })
        
        if not self.is_buff_active("Ice Armor") and not self.is_buff_active("Frost Armor"):
            rotation.append({
                "name": "Ice Armor",
                "type": "buff",
                "target": "self",
                "priority": 8.5,
                "mana_cost": 400
            })
        
        # Check if we need to use defensive abilities
        health_percent = self.get_resource_percent("health")
        
        if health_percent < 40 and not self.is_ability_on_cooldown("Ice Block"):
            rotation.append({
                "name": "Ice Block",
                "type": "defensive",
                "target": "self",
                "priority": 10.0,
                "mana_cost": 0
            })
        
        # Check for crowd control if multiple enemies
        if hasattr(state, "nearby_entities"):
            nearby_enemies = [e for e in state.nearby_entities 
                             if e.get("hostile", False) and e.get("id") != self.current_target]
            
            # If we have multiple enemies and one is not controlled
            if (len(nearby_enemies) > 0 and 
                not any(e.get("id") in self.polymorph_targets for e in nearby_enemies)):
                
                # Find a suitable CC target
                cc_target = next((e for e in nearby_enemies 
                                  if e.get("type") == "humanoid" or e.get("type") == "beast"), 
                                 nearby_enemies[0])
                
                rotation.append({
                    "name": "Polymorph",
                    "type": "cc",
                    "target": cc_target.get("id"),
                    "priority": 9.5,
                    "mana_cost": 300
                })
                
                # Record the polymorph (would last for 30 seconds)
                self.polymorph_targets[cc_target.get("id")] = time.time() + 30
        
        # Use defensive AOE if surrounded
        if (hasattr(state, "nearby_entities") and 
            len([e for e in state.nearby_entities 
                if e.get("hostile", False) and 
                e.get("distance", 100) < 6]) >= 2):
            
            if not self.is_ability_on_cooldown("Frost Nova"):
                rotation.append({
                    "name": "Frost Nova",
                    "type": "cc",
                    "target": "self",
                    "priority": 9.0,
                    "mana_cost": 200
                })
        
        # Main frost rotation
        
        # Use Brain Freeze proc if available
        if self.has_proc_brain_freeze and not self.is_ability_on_cooldown("Frostfire Bolt"):
            rotation.append({
                "name": "Frostfire Bolt",
                "type": "damage",
                "target": self.current_target,
                "priority": 8.7,
                "mana_cost": 0  # Free cast
            })
        
        # Use Fingers of Frost proc if available
        if self.has_proc_fingers_of_frost and not self.is_ability_on_cooldown("Ice Lance"):
            rotation.append({
                "name": "Ice Lance",
                "type": "damage",
                "target": self.current_target,
                "priority": 8.5,
                "mana_cost": 100
            })
        
        # Deep Freeze for burst damage (if target is frozen)
        if (not self.is_ability_on_cooldown("Deep Freeze") and 
            any(debuff in ["Frost Nova", "Freeze"] for debuff in self.active_frost_debuffs)):
            rotation.append({
                "name": "Deep Freeze",
                "type": "damage",
                "target": self.current_target,
                "priority": 8.3,
                "mana_cost": 300
            })
        
        # Use cooldowns for burst damage
        if not self.is_ability_on_cooldown("Icy Veins"):
            rotation.append({
                "name": "Icy Veins",
                "type": "cooldown",
                "target": "self",
                "priority": 7.0,
                "mana_cost": 0
            })
        
        # Water Elemental if available and not already summoned
        if (not self.is_ability_on_cooldown("Summon Water Elemental") and 
            not self.is_buff_active("Water Elemental")):
            rotation.append({
                "name": "Summon Water Elemental",
                "type": "summon",
                "target": "self",
                "priority": 7.5,
                "mana_cost": 400
            })
        
        # Use Frozen Orb for AOE and procs if multiple enemies
        if (not self.is_ability_on_cooldown("Frozen Orb") and 
            hasattr(state, "nearby_entities") and 
            len([e for e in state.nearby_entities if e.get("hostile", False)]) >= 2):
            
            rotation.append({
                "name": "Frozen Orb",
                "type": "aoe",
                "target": self.current_target,
                "priority": 8.0,
                "mana_cost": 500
            })
        
        # Frostbolt as main filler damage ability
        rotation.append({
            "name": "Frostbolt",
            "type": "damage",
            "target": self.current_target,
            "priority": 6.0,
            "mana_cost": 200
        })
        
        # Basic Attack as last resort (wand)
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "mana_cost": 0
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def _get_fire_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal rotation for Fire mages
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Check if we need to apply buffs
        if not self.is_buff_active("Arcane Intellect"):
            rotation.append({
                "name": "Arcane Intellect",
                "type": "buff",
                "target": "self",
                "priority": 9.0,
                "mana_cost": 500
            })
        
        if not self.is_buff_active("Molten Armor"):
            rotation.append({
                "name": "Molten Armor",
                "type": "buff",
                "target": "self",
                "priority": 8.5,
                "mana_cost": 400
            })
        
        # Check if we need to use defensive abilities
        health_percent = self.get_resource_percent("health")
        
        if health_percent < 30 and not self.is_ability_on_cooldown("Ice Block"):
            rotation.append({
                "name": "Ice Block",
                "type": "defensive",
                "target": "self",
                "priority": 10.0,
                "mana_cost": 0
            })
        
        # Use defensive AOE if surrounded
        if (hasattr(state, "nearby_entities") and 
            len([e for e in state.nearby_entities 
                if e.get("hostile", False) and 
                e.get("distance", 100) < 6]) >= 2):
            
            if not self.is_ability_on_cooldown("Dragon's Breath"):
                rotation.append({
                    "name": "Dragon's Breath",
                    "type": "cc",
                    "target": "self",
                    "priority": 9.0,
                    "mana_cost": 300
                })
        
        # Main fire rotation
        
        # Use Hot Streak proc for instant Pyroblast
        if self.has_proc_hot_streak:
            rotation.append({
                "name": "Pyroblast",
                "type": "damage",
                "target": self.current_target,
                "priority": 9.0,
                "mana_cost": 0  # Free with Hot Streak
            })
        
        # Use Living Bomb if it's not on the target
        if not self.is_debuff_on_target("Living Bomb") and not self.is_ability_on_cooldown("Living Bomb"):
            rotation.append({
                "name": "Living Bomb",
                "type": "damage",
                "target": self.current_target,
                "priority": 8.0,
                "mana_cost": 300
            })
        
        # Use cooldowns for burst damage
        if not self.is_ability_on_cooldown("Combustion"):
            # Ideally use with multiple DOTs active
            if len(self.active_fire_debuffs) >= 2:
                rotation.append({
                    "name": "Combustion",
                    "type": "cooldown",
                    "target": self.current_target,
                    "priority": 8.5,
                    "mana_cost": 0
                })
        
        # Use AOE for multiple enemies
        if (hasattr(state, "nearby_entities") and 
            len([e for e in state.nearby_entities if e.get("hostile", False)]) >= 3):
            
            if not self.is_ability_on_cooldown("Flamestrike"):
                rotation.append({
                    "name": "Flamestrike",
                    "type": "aoe",
                    "target": self.current_target,
                    "priority": 7.5,
                    "mana_cost": 600
                })
            
            if not self.is_ability_on_cooldown("Blast Wave"):
                rotation.append({
                    "name": "Blast Wave",
                    "type": "aoe",
                    "target": "self",
                    "priority": 7.0,
                    "mana_cost": 500
                })
        
        # Fireball as main filler damage ability
        rotation.append({
            "name": "Fireball",
            "type": "damage",
            "target": self.current_target,
            "priority": 6.0,
            "mana_cost": 300
        })
        
        # Fire Blast for instant damage (especially when moving)
        if not self.is_ability_on_cooldown("Fire Blast"):
            rotation.append({
                "name": "Fire Blast",
                "type": "damage",
                "target": self.current_target,
                "priority": 5.0,
                "mana_cost": 250
            })
        
        # Scorch as low-mana alternative
        if self.get_resource_percent("mana") < 30:
            rotation.append({
                "name": "Scorch",
                "type": "damage",
                "target": self.current_target,
                "priority": 5.5,
                "mana_cost": 150
            })
        
        # Basic Attack as last resort (wand)
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "mana_cost": 0
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def _get_arcane_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal rotation for Arcane mages
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Check if we need to apply buffs
        if not self.is_buff_active("Arcane Intellect"):
            rotation.append({
                "name": "Arcane Intellect",
                "type": "buff",
                "target": "self",
                "priority": 9.0,
                "mana_cost": 500
            })
        
        if not self.is_buff_active("Mage Armor"):
            rotation.append({
                "name": "Mage Armor",
                "type": "buff",
                "target": "self",
                "priority": 8.5,
                "mana_cost": 400
            })
        
        # Check for enemy spell casting
        is_enemy_casting = False
        if hasattr(state, "target_casting") and state.target_casting:
            is_enemy_casting = True
        
        # Main arcane rotation
        
        # Get current mana percentage
        mana_percent = self.get_resource_percent("mana")
        
        # Get arcane blast stacks
        arcane_blast_stacks = 0
        if hasattr(state, "player_buffs") and "Arcane Blast" in state.player_buffs:
            # In a real implementation we'd get the actual count
            arcane_blast_stacks = state.player_buffs["Arcane Blast"].get("stacks", 0)
        
        # Use Counterspell to interrupt enemy caster
        if is_enemy_casting and not self.is_ability_on_cooldown("Counterspell"):
            rotation.append({
                "name": "Counterspell",
                "type": "interrupt",
                "target": self.current_target,
                "priority": 10.0,
                "mana_cost": 200
            })
        
        # Use cooldowns for burst damage
        if not self.is_ability_on_cooldown("Arcane Power"):
            if mana_percent > 80:  # Only use when high on mana
                rotation.append({
                    "name": "Arcane Power",
                    "type": "cooldown",
                    "target": "self",
                    "priority": 8.0,
                    "mana_cost": 0
                })
        
        # Use Mirror Image for threat management if available
        if not self.is_ability_on_cooldown("Mirror Image"):
            rotation.append({
                "name": "Mirror Image",
                "type": "cooldown",
                "target": "self",
                "priority": 7.5,
                "mana_cost": 300
            })
        
        # Use AOE for multiple enemies
        if (hasattr(state, "nearby_entities") and 
            len([e for e in state.nearby_entities if e.get("hostile", False)]) >= 3):
            
            if not self.is_ability_on_cooldown("Arcane Explosion"):
                rotation.append({
                    "name": "Arcane Explosion",
                    "type": "aoe",
                    "target": "self",
                    "priority": 7.0,
                    "mana_cost": 400
                })
        
        # Arcane Barrage to reset Arcane Blast stacks when mana gets low
        if mana_percent < 50 and arcane_blast_stacks >= 3 and not self.is_ability_on_cooldown("Arcane Barrage"):
            rotation.append({
                "name": "Arcane Barrage",
                "type": "damage",
                "target": self.current_target,
                "priority": 8.0,
                "mana_cost": 200
            })
        
        # Evocation to restore mana when very low
        if mana_percent < 20 and not self.is_ability_on_cooldown("Evocation"):
            rotation.append({
                "name": "Evocation",
                "type": "resource",
                "target": "self",
                "priority": 9.0,
                "mana_cost": 0
            })
        
        # Arcane Missiles when Missile Barrage procs
        if self.is_buff_active("Missile Barrage") and not self.is_ability_on_cooldown("Arcane Missiles"):
            rotation.append({
                "name": "Arcane Missiles",
                "type": "damage",
                "target": self.current_target,
                "priority": 8.5,
                "mana_cost": 100  # Reduced with proc
            })
        
        # Arcane Blast as main damage ability when mana is sufficient
        if mana_percent > 30:
            rotation.append({
                "name": "Arcane Blast",
                "type": "damage",
                "target": self.current_target,
                "priority": 6.0,
                "mana_cost": 300  # Increases with stacks
            })
        
        # Arcane Missiles as a filler when mana is lower
        if mana_percent < 50 and not self.is_ability_on_cooldown("Arcane Missiles"):
            rotation.append({
                "name": "Arcane Missiles",
                "type": "damage",
                "target": self.current_target,
                "priority": 5.0,
                "mana_cost": 400
            })
        
        # Basic Attack as last resort (wand)
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "mana_cost": 0
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def _get_generic_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get a generic mage rotation (for low levels or when talent build is unknown)
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Ability rotation
        """
        rotation = []
        
        # Check if we need to apply buffs
        if not self.is_buff_active("Arcane Intellect"):
            rotation.append({
                "name": "Arcane Intellect",
                "type": "buff",
                "target": "self",
                "priority": 9.0,
                "mana_cost": 500
            })
        
        if not self.is_buff_active("Frost Armor"):
            rotation.append({
                "name": "Frost Armor",
                "type": "buff",
                "target": "self",
                "priority": 8.5,
                "mana_cost": 300
            })
        
        # Get current mana percentage
        mana_percent = self.get_resource_percent("mana")
        
        # Use defensive abilities if surrounded
        if (hasattr(state, "nearby_entities") and 
            len([e for e in state.nearby_entities 
                if e.get("hostile", False) and 
                e.get("distance", 100) < 6]) >= 1):
            
            if not self.is_ability_on_cooldown("Frost Nova"):
                rotation.append({
                    "name": "Frost Nova",
                    "type": "cc",
                    "target": "self",
                    "priority": 9.0,
                    "mana_cost": 200
                })
            
            # Blink away if available
            if not self.is_ability_on_cooldown("Blink"):
                rotation.append({
                    "name": "Blink",
                    "type": "escape",
                    "target": "self",
                    "priority": 8.5,
                    "mana_cost": 200
                })
        
        # Basic damage rotation
        
        # Fire Blast for instant damage if available
        if not self.is_ability_on_cooldown("Fire Blast"):
            rotation.append({
                "name": "Fire Blast",
                "type": "damage",
                "target": self.current_target,
                "priority": 7.0,
                "mana_cost": 250
            })
        
        # Choose main damage spell based on mana
        if mana_percent > 50:
            rotation.append({
                "name": "Frostbolt",
                "type": "damage",
                "target": self.current_target,
                "priority": 6.0,
                "mana_cost": 200
            })
        else:
            rotation.append({
                "name": "Arcane Missiles",
                "type": "damage",
                "target": self.current_target,
                "priority": 5.5,
                "mana_cost": 400
            })
        
        # Basic Attack as last resort (wand)
        rotation.append({
            "name": "Attack",
            "type": "basic",
            "target": self.current_target,
            "priority": 1.0,
            "mana_cost": 0
        })
        
        # Sort by priority (highest first)
        rotation.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return rotation
    
    def get_optimal_target(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Get the optimal target for a mage
        
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
                    objective_text = objective.get("text", "")
                    if "kill" in objective_text.lower():
                        # Extract target name (simple heuristic)
                        parts = objective_text.split()
                        for i, part in enumerate(parts):
                            if part.lower() == "kill" and i + 1 < len(parts):
                                quest_targets.append(parts[i + 1])
        
        # Check for polymorphed targets (avoid targeting these)
        polymorphed_targets = set(self.polymorph_targets.keys())
        
        # Filter nearby entities
        suitable_targets = []
        
        for entity in state.nearby_entities:
            entity_name = entity.get("name", "")
            entity_id = entity.get("id", "")
            is_hostile = entity.get("hostile", False)
            
            # Skip non-hostile entities and polymorphed targets
            if not is_hostile or entity_id in polymorphed_targets:
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
            
            # Adjust score based on distance (prefer optimal range)
            if distance > 0:
                if 15 <= distance <= 30:
                    score += 15  # Optimal range for mages
                elif distance < 10:
                    score -= 20  # Too close
                elif distance > 35:
                    score -= 10  # Too far
            
            # Adjust score based on enemy type (prefer casters for interrupts)
            if entity.get("class") in ["mage", "priest", "warlock", "shaman", "druid"]:
                score += 10  # Prioritize other casters
            
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
        Get the optimal position for a mage in combat
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Tuple]: Position coordinates or None
        """
        if not self.current_target or "position" not in self.target_data:
            return None
        
        # Get target position
        target_x, target_y = self.target_data["position"]
        
        # Mages want to be at medium-to-long range
        optimal_range = 25.0
        
        # Get current distance to target
        current_distance = self.calculate_distance_to_target()
        
        # If we're at a good range, no need to move
        if 20 <= current_distance <= 30:
            return None
        
        # Check if we're too close
        if current_distance < 15:
            # Try to get further away
            # Move directly away from the target
            if hasattr(self, "player_position") and self.player_position:
                px, py = self.player_position
                
                # Calculate direction vector
                dx = px - target_x
                dy = py - target_y
                
                # Normalize and scale
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx = dx / length * optimal_range
                    dy = dy / length * optimal_range
                    
                    # New position further away from the target
                    return (target_x + dx, target_y + dy)
        
        # Otherwise, generate position at optimal range
        # Add a slight random offset to avoid being perfectly predictable
        angle = random.uniform(0, 2 * math.pi)
        offset_x = optimal_range * math.cos(angle)
        offset_y = optimal_range * math.sin(angle)
        
        return (target_x + offset_x, target_y + offset_y)
    
    def get_resource_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get abilities for resource management (mana regeneration)
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of resource management abilities
        """
        resource_abilities = []
        
        # Check current mana
        mana_percent = self.get_resource_percent("mana")
        
        # Evocation for major mana regeneration
        if mana_percent < 30 and not self.is_ability_on_cooldown("Evocation"):
            resource_abilities.append({
                "name": "Evocation",
                "type": "ability",
                "target": "self",
                "priority": 9.0,
                "description": "Restore mana"
            })
        
        # Use mana gem if available
        if mana_percent < 60 and not self.is_ability_on_cooldown("Mana Gem"):
            resource_abilities.append({
                "name": "Mana Gem",
                "type": "item",
                "target": "self",
                "priority": 7.0,
                "description": "Use mana gem to restore mana"
            })
        
        return resource_abilities
    
    def get_defensive_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get defensive abilities for mages
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of defensive abilities
        """
        defensive_abilities = []
        
        # Check health percentage
        health_percent = self.get_resource_percent("health")
        
        # Ice Block for emergency situations
        if health_percent < 25 and not self.is_ability_on_cooldown("Ice Block"):
            defensive_abilities.append({
                "name": "Ice Block",
                "type": "defensive",
                "target": "self",
                "priority": 10.0,
                "description": "Become immune to all damage"
            })
        
        # Frost Nova if enemies are close
        if (hasattr(state, "nearby_entities") and 
            len([e for e in state.nearby_entities 
                if e.get("hostile", False) and 
                e.get("distance", 100) < 8]) > 0 and
            not self.is_ability_on_cooldown("Frost Nova")):
            
            defensive_abilities.append({
                "name": "Frost Nova",
                "type": "defensive",
                "target": "self",
                "priority": 9.0,
                "description": "Freeze nearby enemies"
            })
        
        # Blink to escape
        if health_percent < 40 and not self.is_ability_on_cooldown("Blink"):
            defensive_abilities.append({
                "name": "Blink",
                "type": "defensive",
                "target": "self",
                "priority": 8.5,
                "description": "Teleport forward and break snares"
            })
        
        # Mana Shield as a last resort
        if health_percent < 30 and not self.is_buff_active("Mana Shield") and self.get_resource_percent("mana") > 30:
            defensive_abilities.append({
                "name": "Mana Shield",
                "type": "defensive",
                "target": "self",
                "priority": 8.0,
                "description": "Absorb damage using mana"
            })
        
        return defensive_abilities
    
    def get_supported_talent_builds(self) -> List[Dict[str, Any]]:
        """
        Get supported talent builds for mages
        
        Returns:
            List[Dict]: List of supported talent builds with their rotations
        """
        return [
            {
                "name": "Frost",
                "description": "High control with snares, roots, and consistent damage",
                "key_abilities": ["Frostbolt", "Ice Lance", "Deep Freeze"],
                "strengths": "Excellent kiting and PvP control",
                "weaknesses": "Lower burst damage than Fire"
            },
            {
                "name": "Fire",
                "description": "High burst damage with crits and DOTs",
                "key_abilities": ["Fireball", "Pyroblast", "Living Bomb"],
                "strengths": "Best AoE and burst damage",
                "weaknesses": "More vulnerable to interruption"
            },
            {
                "name": "Arcane",
                "description": "Mana-based damage with high single target DPS",
                "key_abilities": ["Arcane Blast", "Arcane Missiles", "Arcane Barrage"],
                "strengths": "Highest single target damage",
                "weaknesses": "Mana management challenges"
            }
        ]