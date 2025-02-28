"""
Priest Combat Module

This module implements the class-specific combat logic for the Priest class.
"""

import logging
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.combat.base_combat_module import BaseCombatModule


class PriestCombatModule(BaseCombatModule):
    """
    Priest-specific combat module implementing the BaseCombatModule interface.
    
    This module handles combat rotations, resource management, and positioning
    specific to the Priest class in World of Warcraft.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge: GameKnowledge):
        """
        Initialize the Priest combat module
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        super().__init__(config, knowledge)
        
        # Priest-specific state tracking
        self.shadow_form_active = False
        self.inner_fire_active = False
        self.power_word_shield_cooldown = 0
        self.shield_target_id = None
        self.last_healing_target_id = None
        self.current_dot_targets = {}  # Track DoT applications by target
        
        # Add mana resource
        self.current_resources["mana"] = 0
        self.max_resources["mana"] = 100
        
        # Track nearby enemies for AoE decisions
        self.nearby_enemies_count: int = 0
        self.aoe_threshold: int = 3
        
        # Group member tracking
        self.group_members = []
        self.group_members_health = {}  # member_id -> health percent
        
        self.logger.info("PriestCombatModule initialized")
    
    def update_state(self, state: GameState) -> None:
        """
        Update priest-specific state
        
        Args:
            state: Current game state
        """
        # Call parent update first
        super().update_state(state)
        
        # Update priest-specific resources
        self._update_priest_resources(state)
        
        # Update form and buff states
        self._update_priest_buffs(state)
        
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
            
        # Update group member information
        if hasattr(state, "group_members"):
            self.group_members = state.group_members
            
            if hasattr(state, "group_members_health"):
                self.group_members_health = state.group_members_health
    
    def _update_priest_resources(self, state: GameState) -> None:
        """
        Update priest-specific resources like mana
        
        Args:
            state: Current game state
        """
        # Get mana value if available
        if hasattr(state, "player_mana"):
            self.current_resources["mana"] = state.player_mana
        
        if hasattr(state, "player_mana_max"):
            self.max_resources["mana"] = state.player_mana_max
    
    def _update_priest_buffs(self, state: GameState) -> None:
        """
        Update priest buff states like shadowform
        
        Args:
            state: Current game state
        """
        # Reset buff states
        self.shadow_form_active = False
        self.inner_fire_active = False
        
        # Check player buffs
        if hasattr(state, "player_buffs"):
            for buff_name in state.player_buffs:
                if buff_name == "Shadowform":
                    self.shadow_form_active = True
                elif buff_name == "Inner Fire":
                    self.inner_fire_active = True
        
        # Update Shield cooldown
        current_time = time.time()
        if self.power_word_shield_cooldown <= current_time:
            self.power_word_shield_cooldown = 0
    
    def get_optimal_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal priest ability rotation based on specialization and state
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        # Determine the spec from talents or config
        spec = self._determine_specialization(state)
        
        # In a group, healers should focus on healing
        if self.group_members and spec in ["holy", "discipline"]:
            return self._get_group_healing_rotation(state)
        
        # Generate rotation based on spec
        if spec == "shadow":
            return self._get_shadow_rotation(state)
        elif spec == "discipline":
            return self._get_discipline_rotation(state)
        elif spec == "holy":
            return self._get_holy_rotation(state)
        else:
            return self._get_leveling_rotation(state)
    
    def _determine_specialization(self, state: GameState) -> str:
        """
        Determine priest specialization from talents
        
        Args:
            state: Current game state
            
        Returns:
            str: Specialization name (shadow, discipline, holy)
        """
        # Check config first
        if "priest_spec" in self.config:
            return self.config["priest_spec"]
        
        # Check for shadowform as a simple indicator
        if self.shadow_form_active:
            return "shadow"
        
        # Default to shadow if no information is available
        return "shadow"
    
    def _get_shadow_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal shadow priest rotation
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Ensure Shadowform is active
        if not self.shadow_form_active:
            rotation.append({
                "name": "Shadowform",
                "target": "self",
                "priority": 100,
                "mana_cost": 30,
                "condition": "form not active"
            })
        
        # Apply Power Word: Shield if not on cooldown and not already shielded
        if not self.power_word_shield_cooldown and not self.is_buff_active("Power Word: Shield"):
            rotation.append({
                "name": "Power Word: Shield",
                "target": "self",
                "priority": 95,
                "mana_cost": 20,
                "condition": "not on cooldown"
            })
            self.power_word_shield_cooldown = time.time() + 15  # 15-second cooldown
        
        # Use Shadowfiend for mana when below 50%
        if self.get_resource_percent("mana") < 50 and not self.is_ability_on_cooldown("Shadowfiend"):
            rotation.append({
                "name": "Shadowfiend",
                "target": self.current_target,
                "priority": 90,
                "mana_cost": 0,
                "condition": "mana < 50%"
            })
        
        # AoE rotation if multiple enemies
        if self.nearby_enemies_count >= self.aoe_threshold:
            # Mind Sear for AoE
            rotation.append({
                "name": "Mind Sear",
                "target": self.current_target,
                "priority": 85,
                "mana_cost": 25,
                "condition": "nearby_enemies >= 3"
            })
        
        # Maintain DoTs
        if not self.is_debuff_on_target("Shadow Word: Pain"):
            rotation.append({
                "name": "Shadow Word: Pain",
                "target": self.current_target,
                "priority": 80,
                "mana_cost": 15,
                "condition": "debuff not on target"
            })
        
        if not self.is_debuff_on_target("Vampiric Touch"):
            rotation.append({
                "name": "Vampiric Touch",
                "target": self.current_target,
                "priority": 75,
                "mana_cost": 15,
                "condition": "debuff not on target"
            })
        
        if not self.is_debuff_on_target("Devouring Plague"):
            rotation.append({
                "name": "Devouring Plague",
                "target": self.current_target,
                "priority": 70,
                "mana_cost": 15,
                "condition": "debuff not on target"
            })
        
        # Use Mind Blast on cooldown
        if not self.is_ability_on_cooldown("Mind Blast"):
            rotation.append({
                "name": "Mind Blast",
                "target": self.current_target,
                "priority": 65,
                "mana_cost": 15,
                "condition": "not on cooldown"
            })
        
        # Shadow Word: Death for execute phase
        if self.target_data and self.target_data.get("health_percent", 100) < 20:
            rotation.append({
                "name": "Shadow Word: Death",
                "target": self.current_target,
                "priority": 60,
                "mana_cost": 10,
                "condition": "target_health < 20%"
            })
        
        # Mind Flay as filler
        rotation.append({
            "name": "Mind Flay",
            "target": self.current_target,
            "priority": 55,
            "mana_cost": 10,
            "condition": "always"
        })
        
        # Smite as backup
        rotation.append({
            "name": "Smite",
            "target": self.current_target,
            "priority": 50,
            "mana_cost": 10,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_discipline_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal discipline priest rotation for solo play
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Ensure Inner Fire is active
        if not self.inner_fire_active:
            rotation.append({
                "name": "Inner Fire",
                "target": "self",
                "priority": 100,
                "mana_cost": 15,
                "condition": "buff not active"
            })
        
        # Apply Power Word: Shield if not on cooldown and not already shielded
        if not self.power_word_shield_cooldown and not self.is_buff_active("Power Word: Shield"):
            rotation.append({
                "name": "Power Word: Shield",
                "target": "self",
                "priority": 95,
                "mana_cost": 20,
                "condition": "not on cooldown"
            })
            self.power_word_shield_cooldown = time.time() + 15  # 15-second cooldown
        
        # Apply Pain Suppression if health critically low
        if self.get_resource_percent("health") < 20 and not self.is_ability_on_cooldown("Pain Suppression"):
            rotation.append({
                "name": "Pain Suppression",
                "target": "self",
                "priority": 90,
                "mana_cost": 10,
                "condition": "health < 20%"
            })
        
        # Self-healing if health is low
        if self.get_resource_percent("health") < 50:
            rotation.append({
                "name": "Penance",
                "target": "self",
                "priority": 85,
                "mana_cost": 20,
                "condition": "health < 50%"
            })
            
            rotation.append({
                "name": "Flash Heal",
                "target": "self",
                "priority": 80,
                "mana_cost": 15,
                "condition": "health < 50%"
            })
        
        # Maintain DoTs
        if not self.is_debuff_on_target("Shadow Word: Pain"):
            rotation.append({
                "name": "Shadow Word: Pain",
                "target": self.current_target,
                "priority": 75,
                "mana_cost": 15,
                "condition": "debuff not on target"
            })
        
        # Use Penance for damage
        if not self.is_ability_on_cooldown("Penance") and self.get_resource_percent("health") >= 50:
            rotation.append({
                "name": "Penance",
                "target": self.current_target,
                "priority": 70,
                "mana_cost": 20,
                "condition": "not on cooldown"
            })
        
        # Use Mind Blast on cooldown
        if not self.is_ability_on_cooldown("Mind Blast"):
            rotation.append({
                "name": "Mind Blast",
                "target": self.current_target,
                "priority": 65,
                "mana_cost": 15,
                "condition": "not on cooldown"
            })
        
        # Smite as filler
        rotation.append({
            "name": "Smite",
            "target": self.current_target,
            "priority": 60,
            "mana_cost": 10,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_holy_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get the optimal holy priest rotation for solo play
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Ensure Inner Fire is active
        if not self.inner_fire_active:
            rotation.append({
                "name": "Inner Fire",
                "target": "self",
                "priority": 100,
                "mana_cost": 15,
                "condition": "buff not active"
            })
        
        # Apply Power Word: Shield if not on cooldown and not already shielded
        if not self.power_word_shield_cooldown and not self.is_buff_active("Power Word: Shield"):
            rotation.append({
                "name": "Power Word: Shield",
                "target": "self",
                "priority": 95,
                "mana_cost": 20,
                "condition": "not on cooldown"
            })
            self.power_word_shield_cooldown = time.time() + 15  # 15-second cooldown
        
        # Self-healing if health is low
        if self.get_resource_percent("health") < 50:
            rotation.append({
                "name": "Holy Word: Serenity",
                "target": "self",
                "priority": 90,
                "mana_cost": 20,
                "condition": "health < 50%"
            })
            
            rotation.append({
                "name": "Flash Heal",
                "target": "self",
                "priority": 85,
                "mana_cost": 15,
                "condition": "health < 50%"
            })
        
        # Keep Renew up on self
        if not self.is_buff_active("Renew"):
            rotation.append({
                "name": "Renew",
                "target": "self",
                "priority": 80,
                "mana_cost": 15,
                "condition": "buff not active"
            })
        
        # Maintain DoTs
        if not self.is_debuff_on_target("Shadow Word: Pain"):
            rotation.append({
                "name": "Shadow Word: Pain",
                "target": self.current_target,
                "priority": 75,
                "mana_cost": 15,
                "condition": "debuff not on target"
            })
        
        # Holy Fire on cooldown
        if not self.is_ability_on_cooldown("Holy Fire"):
            rotation.append({
                "name": "Holy Fire",
                "target": self.current_target,
                "priority": 70,
                "mana_cost": 15,
                "condition": "not on cooldown"
            })
        
        # Holy Word: Chastise for control
        if not self.is_ability_on_cooldown("Holy Word: Chastise"):
            rotation.append({
                "name": "Holy Word: Chastise",
                "target": self.current_target,
                "priority": 65,
                "mana_cost": 15,
                "condition": "not on cooldown"
            })
        
        # Smite as filler
        rotation.append({
            "name": "Smite",
            "target": self.current_target,
            "priority": 60,
            "mana_cost": 10,
            "condition": "always"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_group_healing_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get healing rotation for group content
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of healing abilities in priority order
        """
        rotation = []
        
        # Ensure Inner Fire/Will is active
        if not self.inner_fire_active:
            rotation.append({
                "name": "Inner Fire",
                "target": "self",
                "priority": 100,
                "mana_cost": 15,
                "condition": "buff not active"
            })
        
        # Find lowest health group member
        lowest_health_member = self._get_lowest_health_member()
        
        # Critical healing for very low health members
        if lowest_health_member and self.group_members_health.get(lowest_health_member, 100) < 30:
            # Guardian Spirit to prevent death
            if not self.is_ability_on_cooldown("Guardian Spirit") and self._determine_specialization(state) == "holy":
                rotation.append({
                    "name": "Guardian Spirit",
                    "target": lowest_health_member,
                    "priority": 95,
                    "mana_cost": 15,
                    "condition": "health < 30%"
                })
            
            # Pain Suppression for damage reduction
            if not self.is_ability_on_cooldown("Pain Suppression") and self._determine_specialization(state) == "discipline":
                rotation.append({
                    "name": "Pain Suppression",
                    "target": lowest_health_member,
                    "priority": 95,
                    "mana_cost": 15,
                    "condition": "health < 30%"
                })
            
            # Flash Heal for emergency healing
            rotation.append({
                "name": "Flash Heal",
                "target": lowest_health_member,
                "priority": 90,
                "mana_cost": 20,
                "condition": "health < 30%"
            })
        
        # Group healing if multiple members are injured
        injured_count = sum(1 for health in self.group_members_health.values() if health < 80)
        if injured_count >= 3:
            if self._determine_specialization(state) == "holy":
                rotation.append({
                    "name": "Holy Word: Sanctify",
                    "target": "none",  # AoE heal at target location
                    "priority": 85,
                    "mana_cost": 25,
                    "condition": "multiple injured"
                })
                
                rotation.append({
                    "name": "Prayer of Healing",
                    "target": "none",  # Group heal
                    "priority": 80,
                    "mana_cost": 30,
                    "condition": "multiple injured"
                })
            
            # Circle of Healing for quick AoE heal
            if not self.is_ability_on_cooldown("Circle of Healing"):
                rotation.append({
                    "name": "Circle of Healing",
                    "target": "none",  # AoE heal
                    "priority": 75,
                    "mana_cost": 20,
                    "condition": "multiple injured"
                })
            
            # Divine Hymn for serious group damage
            if not self.is_ability_on_cooldown("Divine Hymn") and injured_count >= 4:
                rotation.append({
                    "name": "Divine Hymn",
                    "target": "none",  # Channel group heal
                    "priority": 70,
                    "mana_cost": 35,
                    "condition": "multiple severely injured"
                })
        
        # Regular healing for moderately injured members
        if lowest_health_member and self.group_members_health.get(lowest_health_member, 100) < 70:
            # Discipline specific
            if self._determine_specialization(state) == "discipline":
                if not self.is_ability_on_cooldown("Penance") and self.group_members_health.get(lowest_health_member, 100) < 60:
                    rotation.append({
                        "name": "Penance",
                        "target": lowest_health_member,
                        "priority": 65,
                        "mana_cost": 20,
                        "condition": "health < 60%"
                    })
                
                if not self.power_word_shield_cooldown:
                    rotation.append({
                        "name": "Power Word: Shield",
                        "target": lowest_health_member,
                        "priority": 60,
                        "mana_cost": 20,
                        "condition": "not on cooldown"
                    })
                    self.power_word_shield_cooldown = time.time() + 15  # 15-second cooldown
            
            # Holy specific
            if self._determine_specialization(state) == "holy":
                if not self.is_ability_on_cooldown("Holy Word: Serenity") and self.group_members_health.get(lowest_health_member, 100) < 60:
                    rotation.append({
                        "name": "Holy Word: Serenity",
                        "target": lowest_health_member,
                        "priority": 65,
                        "mana_cost": 20,
                        "condition": "health < 60%"
                    })
            
            # Generic healing
            rotation.append({
                "name": "Heal",
                "target": lowest_health_member,
                "priority": 55,
                "mana_cost": 15,
                "condition": "health < 70%"
            })
            
            # Apply Renew if not already active
            # This would require tracking who has Renew active
            rotation.append({
                "name": "Renew",
                "target": lowest_health_member,
                "priority": 50,
                "mana_cost": 15,
                "condition": "health < 80% and no renew"
            })
        
        # Mana recovery
        if self.get_resource_percent("mana") < 30:
            rotation.append({
                "name": "Shadowfiend",
                "target": self.current_target or "target",
                "priority": 45,
                "mana_cost": 0,
                "condition": "mana < 30%"
            })
            
            rotation.append({
                "name": "Hymn of Hope",
                "target": "self",
                "priority": 40,
                "mana_cost": 0,
                "condition": "mana < 20%"
            })
        
        # Apply Fortitude if anyone is missing it
        rotation.append({
            "name": "Power Word: Fortitude",
            "target": "group",
            "priority": 35,
            "mana_cost": 20,
            "condition": "missing buff"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def _get_lowest_health_member(self) -> Optional[str]:
        """
        Get the group member with the lowest health
        
        Returns:
            Optional[str]: Member ID or None
        """
        if not self.group_members_health:
            return None
        
        return min(self.group_members_health.items(), key=lambda x: x[1])[0]
    
    def _get_leveling_rotation(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get a simple leveling rotation for priests without many abilities
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of abilities to use in order
        """
        rotation = []
        
        # Apply Power Word: Shield if not on cooldown and not already shielded
        if not self.power_word_shield_cooldown and not self.is_buff_active("Power Word: Shield"):
            rotation.append({
                "name": "Power Word: Shield",
                "target": "self",
                "priority": 100,
                "mana_cost": 20,
                "condition": "not on cooldown"
            })
            self.power_word_shield_cooldown = time.time() + 15  # 15-second cooldown
        
        # Self-healing if health is low
        if self.get_resource_percent("health") < 60:
            rotation.append({
                "name": "Renew",
                "target": "self",
                "priority": 90,
                "mana_cost": 15,
                "condition": "health < 60% and no renew"
            })
            
            if self.get_resource_percent("health") < 40:
                rotation.append({
                    "name": "Flash Heal",
                    "target": "self",
                    "priority": 85,
                    "mana_cost": 15,
                    "condition": "health < 40%"
                })
        
        # Apply Shadow Word: Pain for DoT
        if not self.is_debuff_on_target("Shadow Word: Pain"):
            rotation.append({
                "name": "Shadow Word: Pain",
                "target": self.current_target,
                "priority": 80,
                "mana_cost": 15,
                "condition": "debuff not on target"
            })
        
        # Use special abilities as they become available
        abilities_to_check = ["Mind Blast", "Smite", "Mind Flay"]
        for ability in abilities_to_check:
            if ability == "Mind Blast" and not self.is_ability_on_cooldown(ability):
                rotation.append({
                    "name": ability,
                    "target": self.current_target,
                    "priority": 70,
                    "mana_cost": 15,
                    "condition": "not on cooldown"
                })
            elif ability in self.knowledge.get_available_abilities("priest"):
                rotation.append({
                    "name": ability,
                    "target": self.current_target,
                    "priority": 60,
                    "mana_cost": 10,
                    "condition": "always"
                })
        
        # Wand as mana-free damage
        rotation.append({
            "name": "Shoot",
            "target": self.current_target,
            "priority": 50,
            "mana_cost": 0,
            "condition": "mana < 40%"
        })
        
        # Sort by priority
        rotation.sort(key=lambda x: x["priority"], reverse=True)
        
        return rotation
    
    def get_optimal_target(self, state: GameState) -> Optional[Dict[str, Any]]:
        """
        Get the optimal target for a priest
        
        Args:
            state: Current game state
            
        Returns:
            Optional[Dict]: Target information or None
        """
        # For healing specs in a group, prioritize healing targets
        spec = self._determine_specialization(state)
        if self.group_members and spec in ["holy", "discipline"]:
            # Return None for combat target, let healing rotation handle targets
            return None
        
        # Get all possible targets
        potential_targets = []
        
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                # Check if it's a targetable entity
                if entity.get("type") in ["mob", "player"] and entity.get("reaction") == "hostile":
                    potential_targets.append(entity)
        
        if not potential_targets:
            return None
        
        # Sort targets based on specialization
        if spec == "shadow":
            # Shadow should prioritize targets that already have DoTs applied
            potential_targets.sort(key=lambda e: (
                not any(debuff in str(e.get("debuffs", {})) for debuff in ["Shadow Word: Pain", "Vampiric Touch"]),
                not any(qt.lower() in e.get("id", "").lower() for qt in self._get_quest_targets(state)),  # Then quest targets
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
        Get the optimal position for a priest in combat
        
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
        
        # Shadow priests should stay at range
        if spec == "shadow":
            # If too close, back up
            if distance < 20:
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
            
            # If already at good range (20-40 yards), don't move
            if 20 <= distance <= 40:
                return None
            
            # If too far, move closer but maintain safe distance
            if distance > 40:
                # Calculate position 30 yards away
                if hasattr(self, "player_position") and self.player_position:
                    px, py = self.player_position
                    
                    # Vector from player to target
                    dx, dy = tx - px, ty - py
                    
                    # Normalize the vector
                    length = (dx**2 + dy**2)**0.5
                    if length > 0:
                        dx, dy = dx/length, dy/length
                        
                        # Position should be about 30 yards away from target
                        desired_distance = 30
                        return (px + dx * (distance - desired_distance), py + dy * (distance - desired_distance))
        
        # For healing specs, attempt to stay with the group
        elif spec in ["holy", "discipline"] and self.group_members:
            # Find the center of the group
            group_positions = []
            for member_id in self.group_members:
                if hasattr(state, f"{member_id}_position"):
                    member_pos = getattr(state, f"{member_id}_position")
                    if member_pos:
                        group_positions.append(member_pos)
            
            if group_positions:
                # Calculate center of group
                avg_x = sum(pos[0] for pos in group_positions) / len(group_positions)
                avg_y = sum(pos[1] for pos in group_positions) / len(group_positions)
                
                # If we're far from the group center, move toward it
                if hasattr(self, "player_position") and self.player_position:
                    px, py = self.player_position
                    group_distance = ((px - avg_x)**2 + (py - avg_y)**2)**0.5
                    
                    if group_distance > 20:  # If more than 20 yards from group center
                        return (avg_x, avg_y)
        
        return None
    
    def get_resource_abilities(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Get abilities that should be used to manage priest mana
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: List of resource management abilities
        """
        abilities = []
        
        # Get mana percentage
        mana_percent = self.get_resource_percent("mana")
        
        # Use Shadowfiend when mana is low
        if mana_percent < 30 and not self.is_ability_on_cooldown("Shadowfiend"):
            abilities.append({
                "name": "Shadowfiend",
                "target": self.current_target or "target",
                "description": "Regenerate mana"
            })
        
        # Use Hymn of Hope when mana is very low
        if mana_percent < 20 and not self.is_ability_on_cooldown("Hymn of Hope"):
            abilities.append({
                "name": "Hymn of Hope",
                "target": "self",
                "description": "Channel for mana"
            })
        
        # Use mana potion when mana is critically low
        if mana_percent < 15 and "Mana Potion" in self.knowledge.get_available_consumables():
            abilities.append({
                "name": "Mana Potion",
                "target": "self",
                "description": "Use mana potion"
            })
        
        # Dispersion for shadow priests
        if self.shadow_form_active and mana_percent < 20 and not self.is_ability_on_cooldown("Dispersion"):
            abilities.append({
                "name": "Dispersion",
                "target": "self",
                "description": "Regenerate mana"
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
        
        # Desperate Prayer when health is critically low
        if health_percent < 20 and not self.is_ability_on_cooldown("Desperate Prayer"):
            defensive_abilities.append({
                "name": "Desperate Prayer",
                "target": "self",
                "description": "Emergency healing"
            })
        
        # Power Word: Shield if not on cooldown
        if not self.power_word_shield_cooldown and not self.is_buff_active("Power Word: Shield"):
            defensive_abilities.append({
                "name": "Power Word: Shield",
                "target": "self",
                "description": "Damage absorption"
            })
            self.power_word_shield_cooldown = time.time() + 15  # Set cooldown
        
        # Fear if enemy is close and health is low
        if self.calculate_distance_to_target() < 10 and health_percent < 40 and not self.is_ability_on_cooldown("Psychic Scream"):
            defensive_abilities.append({
                "name": "Psychic Scream",
                "target": "self",
                "description": "Fear nearby enemies"
            })
        
        # Fade to reduce threat
        if not self.is_ability_on_cooldown("Fade") and self.group_members:
            defensive_abilities.append({
                "name": "Fade",
                "target": "self",
                "description": "Reduce threat"
            })
        
        # Dispersion for shadows
        if self.shadow_form_active and health_percent < 30 and not self.is_ability_on_cooldown("Dispersion"):
            defensive_abilities.append({
                "name": "Dispersion",
                "target": "self",
                "description": "Damage reduction and health regen"
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
        Get the list of supported talent builds for priests
        
        Returns:
            List[Dict]: List of supported talent builds with their rotations
        """
        return [
            {
                "name": "Shadow (DPS)",
                "description": "Shadow priest focusing on DoTs and shadow damage",
                "key_talents": [
                    "Shadow Form",
                    "Vampiric Touch",
                    "Vampiric Embrace",
                    "Mind Flay",
                    "Shadow Word: Death"
                ],
                "rotation_priority": [
                    "Shadow Word: Pain",
                    "Vampiric Touch",
                    "Devouring Plague",
                    "Mind Blast",
                    "Shadow Word: Death (below 20% health)",
                    "Mind Flay"
                ]
            },
            {
                "name": "Discipline (Hybrid)",
                "description": "Discipline priest focusing on shields and damage mitigation",
                "key_talents": [
                    "Power Word: Shield",
                    "Penance",
                    "Pain Suppression",
                    "Divine Aegis",
                    "Power Infusion"
                ],
                "rotation_priority": [
                    "Power Word: Shield",
                    "Penance",
                    "Flash Heal (emergency)",
                    "Shadow Word: Pain",
                    "Smite"
                ]
            },
            {
                "name": "Holy (Healing)",
                "description": "Holy priest focusing on healing and group support",
                "key_talents": [
                    "Circle of Healing",
                    "Guardian Spirit",
                    "Divine Hymn",
                    "Holy Word: Serenity",
                    "Renew"
                ],
                "rotation_priority": [
                    "Prayer of Mending",
                    "Circle of Healing",
                    "Holy Word: Serenity",
                    "Flash Heal (emergency)",
                    "Renew",
                    "Heal",
                    "Prayer of Healing (group healing)"
                ]
            }
        ]