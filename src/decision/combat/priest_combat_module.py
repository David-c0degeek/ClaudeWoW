"""
Priest Combat Module

This module implements the priest-specific combat logic.
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
    Priest-specific combat module implementing the base combat module interface.
    
    Focuses on healing, damage over time, and mana efficiency strategies depending
    on talent specialization.
    """
    
    def __init__(self, config: Dict, knowledge: GameKnowledge):
        """
        Initialize the Priest combat module
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge object
        """
        super().__init__(config, knowledge)
        self.logger = logging.getLogger("wow_ai.decision.combat.priest")
        
        # Priest-specific state tracking
        self.shadow_form_active = False
        self.inner_fire_active = False
        self.power_word_shield_cooldown = 0
        self.shield_target_id = None
        self.last_healing_target_id = None
        self.current_dot_targets = {}  # Track DoT applications by target
        
        self.logger.info("Priest combat module initialized")
    
    def get_optimal_rotation(self, state: GameState, specialization: str = None) -> List[Dict]:
        """
        Get the optimal spell rotation for the priest based on the current game state
        and talent specialization
        
        Args:
            state: Current game state
            specialization: Talent specialization (discipline, holy, shadow)
            
        Returns:
            List of spells in priority order
        """
        if not specialization:
            # Try to detect specialization from talents
            specialization = self._detect_specialization(state)
        
        self._update_priest_state(state)
        
        # Choose the appropriate rotation based on specialization
        if specialization == "shadow":
            return self._get_shadow_rotation(state)
        elif specialization == "discipline":
            return self._get_discipline_rotation(state)
        elif specialization == "holy":
            return self._get_holy_rotation(state)
        else:
            # Generic rotation for low-level priests or unknown spec
            return self._get_generic_rotation(state)
    
    def _update_priest_state(self, state: GameState):
        """
        Update the priest-specific state tracking
        
        Args:
            state: Current game state
        """
        # Check for active buffs
        if hasattr(state, "player_buffs") and state.player_buffs:
            self.shadow_form_active = any(buff.get("name", "").lower() == "shadowform" for buff in state.player_buffs)
            self.inner_fire_active = any(buff.get("name", "").lower() == "inner fire" for buff in state.player_buffs)
        
        # Update cooldown tracking (this would be more accurate with game state, but this is a simple implementation)
        if self.power_word_shield_cooldown > 0:
            self.power_word_shield_cooldown = max(0, self.power_word_shield_cooldown - (time.time() - self.last_update_time))
        
        # Update DoT tracking - remove expired DoTs
        current_time = time.time()
        for target_id in list(self.current_dot_targets.keys()):
            for dot_name in list(self.current_dot_targets[target_id].keys()):
                if current_time > self.current_dot_targets[target_id][dot_name]:
                    del self.current_dot_targets[target_id][dot_name]
            
            # Remove empty target entries
            if not self.current_dot_targets[target_id]:
                del self.current_dot_targets[target_id]
        
        self.last_update_time = time.time()
    
    def _detect_specialization(self, state: GameState) -> str:
        """
        Attempt to detect the priest's specialization based on talents and state
        
        Args:
            state: Current game state
            
        Returns:
            Specialization name (shadow, discipline, holy) or "generic"
        """
        # This would ideally check the talent tree, but for simplicity we'll use some heuristics
        
        # Check for defining abilities
        if self.shadow_form_active:
            return "shadow"
        
        # Check for healing vs damage abilities in action bars
        if hasattr(state, "action_bars") and state.action_bars:
            discipline_abilities = ["power word: shield", "inner focus", "power infusion"]
            holy_abilities = ["holy nova", "circle of healing", "prayer of healing"]
            shadow_abilities = ["mind flay", "vampiric embrace", "silence"]
            
            disc_count = sum(1 for ability in discipline_abilities if any(
                button.get("name", "").lower() == ability for button in state.action_bars))
            
            holy_count = sum(1 for ability in holy_abilities if any(
                button.get("name", "").lower() == ability for button in state.action_bars))
            
            shadow_count = sum(1 for ability in shadow_abilities if any(
                button.get("name", "").lower() == ability for button in state.action_bars))
            
            if shadow_count > disc_count and shadow_count > holy_count:
                return "shadow"
            elif disc_count > holy_count:
                return "discipline"
            elif holy_count > 0:
                return "holy"
        
        # Default to generic
        return "generic"
    
    def _get_shadow_rotation(self, state: GameState) -> List[Dict]:
        """
        Get the shadow priest spell rotation
        
        Args:
            state: Current game state
            
        Returns:
            List of spells in priority order
        """
        rotation = []
        target_id = state.target if hasattr(state, "target") and state.target else None
        player_health_percent = state.player_health_percent if hasattr(state, "player_health_percent") else 100
        target_health_percent = state.target_health_percent if hasattr(state, "target_health_percent") else 100
        
        # Get target-specific DoT info
        target_dots = self.current_dot_targets.get(target_id, {}) if target_id else {}
        
        # Emergency healing (shadow priests still need to heal sometimes)
        if player_health_percent < 30:
            rotation.append({
                "type": "cast",
                "spell": "Flash Heal",
                "target": "player",
                "description": "Emergency self-heal"
            })
        
        # Buff management
        if not self.shadow_form_active:
            rotation.append({
                "type": "cast",
                "spell": "Shadowform",
                "target": "player",
                "description": "Enter Shadowform for increased shadow damage"
            })
        
        if not self.inner_fire_active:
            rotation.append({
                "type": "cast",
                "spell": "Inner Fire",
                "target": "player",
                "description": "Apply Inner Fire for increased armor and spell power"
            })
        
        # DoT management
        if target_id and "shadow_word_pain" not in target_dots:
            rotation.append({
                "type": "cast",
                "spell": "Shadow Word: Pain",
                "target": target_id,
                "description": "Apply Shadow Word: Pain DoT"
            })
            # Track the DoT application
            if target_id not in self.current_dot_targets:
                self.current_dot_targets[target_id] = {}
            self.current_dot_targets[target_id]["shadow_word_pain"] = time.time() + 18  # SWP lasts 18 seconds
        
        if target_id and "vampiric_touch" not in target_dots and player_health_percent < 80:
            rotation.append({
                "type": "cast",
                "spell": "Vampiric Touch",
                "target": target_id,
                "description": "Apply Vampiric Touch DoT for health drain"
            })
            # Track the DoT application
            if target_id not in self.current_dot_targets:
                self.current_dot_targets[target_id] = {}
            self.current_dot_targets[target_id]["vampiric_touch"] = time.time() + 15  # VT lasts 15 seconds
            
        if target_id and "devouring_plague" not in target_dots:
            rotation.append({
                "type": "cast",
                "spell": "Devouring Plague",
                "target": target_id,
                "description": "Apply Devouring Plague DoT"
            })
            # Track the DoT application
            if target_id not in self.current_dot_targets:
                self.current_dot_targets[target_id] = {}
            self.current_dot_targets[target_id]["devouring_plague"] = time.time() + 24  # DP lasts 24 seconds
        
        # Core damage abilities
        if target_id:
            rotation.append({
                "type": "cast",
                "spell": "Mind Blast",
                "target": target_id,
                "description": "High damage mind blast"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Mind Flay",
                "target": target_id,
                "description": "Mind Flay channel"
            })
        
        # Execute phase
        if target_id and target_health_percent < 20:
            rotation.append({
                "type": "cast",
                "spell": "Shadow Word: Death",
                "target": target_id,
                "description": "Execute with Shadow Word: Death"
            })
        
        # Filler
        if target_id:
            rotation.append({
                "type": "cast",
                "spell": "Smite",
                "target": target_id,
                "description": "Smite filler damage"
            })
        
        return rotation
    
    def _get_discipline_rotation(self, state: GameState) -> List[Dict]:
        """
        Get the discipline priest spell rotation
        
        Args:
            state: Current game state
            
        Returns:
            List of spells in priority order
        """
        rotation = []
        target_id = state.target if hasattr(state, "target") and state.target else None
        player_health_percent = state.player_health_percent if hasattr(state, "player_health_percent") else 100
        player_mana_percent = state.player_mana_percent if hasattr(state, "player_mana_percent") else 100
        
        # Group member status - in a real implementation this would check all party members
        # Here we'll assume we need to heal if we're in a group and have a target that's friendly
        is_healing_needed = (hasattr(state, "group_members") and state.group_members and 
                            hasattr(state, "target_is_friendly") and state.target_is_friendly)
        
        # Buff management
        if not self.inner_fire_active:
            rotation.append({
                "type": "cast",
                "spell": "Inner Fire",
                "target": "player",
                "description": "Apply Inner Fire for increased armor and spell power"
            })
        
        # Bubble management - high priority
        if self.power_word_shield_cooldown <= 0:
            # Shield self if health is low
            if player_health_percent < 80:
                rotation.append({
                    "type": "cast",
                    "spell": "Power Word: Shield",
                    "target": "player",
                    "description": "Shield self to prevent damage"
                })
                self.power_word_shield_cooldown = 15  # Global cooldown + Weakened Soul duration
                self.shield_target_id = "player"
            # Shield target if in a healing situation
            elif is_healing_needed:
                rotation.append({
                    "type": "cast",
                    "spell": "Power Word: Shield",
                    "target": target_id,
                    "description": "Shield ally to prevent damage"
                })
                self.power_word_shield_cooldown = 15
                self.shield_target_id = target_id
        
        # Healing
        if player_health_percent < 60:
            # Use Flash Heal for emergency healing
            rotation.append({
                "type": "cast", 
                "spell": "Flash Heal",
                "target": "player",
                "description": "Quick emergency self-heal"
            })
        elif player_health_percent < 80:
            # Use Heal for efficient healing
            rotation.append({
                "type": "cast",
                "spell": "Heal",
                "target": "player",
                "description": "Efficient self-heal"
            })
        
        # Heal others if needed
        if is_healing_needed:
            rotation.append({
                "type": "cast",
                "spell": "Penance",
                "target": target_id,
                "description": "Fast channel heal"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Flash Heal",
                "target": target_id,
                "description": "Fast heal for target"
            })
        
        # Damage abilities for when no healing is needed
        if target_id and not is_healing_needed:
            # Apply Shadow Word: Pain for damage over time
            rotation.append({
                "type": "cast",
                "spell": "Shadow Word: Pain",
                "target": target_id,
                "description": "Apply DoT for damage while healing"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Smite",
                "target": target_id, 
                "description": "Basic damage ability"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Holy Fire",
                "target": target_id,
                "description": "High damage spell with DoT component"
            })
        
        # Mana recovery
        if player_mana_percent < 20:
            rotation.append({
                "type": "cast",
                "spell": "Shadowfiend",
                "target": target_id if target_id and not is_healing_needed else None,
                "description": "Summon Shadowfiend for mana recovery"
            })
            
            rotation.append({
                "type": "cast", 
                "spell": "Hymn of Hope",
                "target": "player",
                "description": "Channel for mana recovery"
            })
        
        return rotation
    
    def _get_holy_rotation(self, state: GameState) -> List[Dict]:
        """
        Get the holy priest spell rotation
        
        Args:
            state: Current game state
            
        Returns:
            List of spells in priority order
        """
        rotation = []
        target_id = state.target if hasattr(state, "target") and state.target else None
        player_health_percent = state.player_health_percent if hasattr(state, "player_health_percent") else 100
        player_mana_percent = state.player_mana_percent if hasattr(state, "player_mana_percent") else 100
        
        # Group status - for a holy priest, we assume healing is the primary focus
        is_in_group = hasattr(state, "group_members") and state.group_members
        is_healing_needed = (is_in_group and hasattr(state, "target_is_friendly") and state.target_is_friendly)
        
        # Buff management
        if not self.inner_fire_active:
            rotation.append({
                "type": "cast",
                "spell": "Inner Fire",
                "target": "player",
                "description": "Apply Inner Fire for increased armor and spell power"
            })
        
        # AoE Healing when in group
        if is_in_group:
            # Check if multiple group members are injured
            is_group_injured = False  # In a real implementation, would check all group members
            
            if is_group_injured:
                rotation.append({
                    "type": "cast",
                    "spell": "Circle of Healing",
                    "target": None,  # Smart targeting
                    "description": "AoE smart healing for group"
                })
                
                rotation.append({
                    "type": "cast",
                    "spell": "Prayer of Healing",
                    "target": "party1",  # Would target the appropriate party member in a real implementation
                    "description": "AoE party healing"
                })
                
                rotation.append({
                    "type": "cast",
                    "spell": "Holy Nova",
                    "target": None,
                    "description": "AoE healing centered on self"
                })
        
        # Self healing
        if player_health_percent < 40:
            rotation.append({
                "type": "cast",
                "spell": "Guardian Spirit",
                "target": "player",
                "description": "Emergency protection and healing bonus"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Flash Heal",
                "target": "player", 
                "description": "Fast emergency heal"
            })
        elif player_health_percent < 70:
            rotation.append({
                "type": "cast",
                "spell": "Renew",
                "target": "player",
                "description": "Efficient HoT on self"
            })
        
        # Target healing
        if is_healing_needed:
            rotation.append({
                "type": "cast",
                "spell": "Guardian Spirit",
                "target": target_id,
                "description": "Emergency protection for target"
            })
            
            if state.target_health_percent < 50:
                rotation.append({
                    "type": "cast",
                    "spell": "Flash Heal",
                    "target": target_id,
                    "description": "Fast emergency heal for target"
                })
            
            rotation.append({
                "type": "cast",
                "spell": "Renew",
                "target": target_id,
                "description": "Efficient HoT on target"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Heal",
                "target": target_id,
                "description": "Efficient single-target heal"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Greater Heal",
                "target": target_id,
                "description": "Big single-target heal"
            })
        
        # Damage abilities for solo play
        if target_id and not is_healing_needed:
            rotation.append({
                "type": "cast",
                "spell": "Holy Fire",
                "target": target_id,
                "description": "High damage with DoT component"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Shadow Word: Pain",
                "target": target_id,
                "description": "DoT for damage while healing"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Holy Nova",
                "target": None,
                "description": "AoE damage and healing"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Smite",
                "target": target_id,
                "description": "Basic damage spell"
            })
        
        # Mana recovery
        if player_mana_percent < 20:
            rotation.append({
                "type": "cast",
                "spell": "Shadowfiend",
                "target": target_id if target_id and not is_healing_needed else None,
                "description": "Summon Shadowfiend for mana recovery"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Hymn of Hope",
                "target": "player",
                "description": "Channel for mana recovery"
            })
        
        return rotation
    
    def _get_generic_rotation(self, state: GameState) -> List[Dict]:
        """
        Get a generic priest rotation for low-level characters without specialization
        
        Args:
            state: Current game state
            
        Returns:
            List of spells in priority order
        """
        rotation = []
        target_id = state.target if hasattr(state, "target") and state.target else None
        player_health_percent = state.player_health_percent if hasattr(state, "player_health_percent") else 100
        
        # Buff management
        if not self.inner_fire_active:
            rotation.append({
                "type": "cast",
                "spell": "Inner Fire",
                "target": "player",
                "description": "Apply Inner Fire for increased armor and spell power"
            })
        
        # Shield if available (always useful)
        if self.power_word_shield_cooldown <= 0:
            rotation.append({
                "type": "cast",
                "spell": "Power Word: Shield",
                "target": "player",
                "description": "Shield self to prevent damage"
            })
            self.power_word_shield_cooldown = 15
            self.shield_target_id = "player"
        
        # Healing
        if player_health_percent < 70:
            rotation.append({
                "type": "cast",
                "spell": "Renew",
                "target": "player",
                "description": "HoT on self"
            })
        
        if player_health_percent < 50:
            rotation.append({
                "type": "cast",
                "spell": "Flash Heal",
                "target": "player",
                "description": "Fast emergency heal"
            })
        
        # Damage
        if target_id:
            rotation.append({
                "type": "cast",
                "spell": "Shadow Word: Pain",
                "target": target_id, 
                "description": "Apply DoT for steady damage"
            })
            
            rotation.append({
                "type": "cast",
                "spell": "Smite",
                "target": target_id,
                "description": "Basic damage spell"
            })
        
        return rotation
    
    def get_optimal_target(self, state: GameState, specialization: str = None) -> Dict:
        """
        Get the optimal target based on the priest's specialization and game state
        
        Args:
            state: Current game state
            specialization: Talent specialization
            
        Returns:
            Target selection information
        """
        if not specialization:
            specialization = self._detect_specialization(state)
        
        # Default target info
        target_info = {
            "id": None,
            "reason": "No suitable target found",
            "score": 0
        }
        
        # No targets available
        if not hasattr(state, "nearby_entities") or not state.nearby_entities:
            return target_info
        
        targets = []
        
        # For healing specs, prioritize injured friendly targets
        if specialization in ["discipline", "holy"]:
            # Check if there are injured group members
            if hasattr(state, "group_members") and state.group_members:
                for member in state.group_members:
                    health_percent = member.get("health_percent", 100)
                    is_tank = member.get("role", "") == "tank"
                    
                    # Score is higher for lower health and for tanks
                    score = (100 - health_percent) * (1.5 if is_tank else 1.0)
                    
                    if health_percent < 90:  # Only consider healing if they're injured
                        targets.append({
                            "id": member.get("id"),
                            "reason": f"Injured group member ({health_percent}% health)",
                            "score": score
                        })
        
        # For shadow or when no healing is needed, prioritize hostile targets
        # Also used for low-health emergencies on healing specs
        hostile_targets = []
        for entity in state.nearby_entities:
            entity_id = entity.get("id")
            entity_type = entity.get("type", "")
            entity_level = entity.get("level", 1)
            player_level = state.player_level if hasattr(state, "player_level") else 1
            
            # Skip friendly targets for damage
            if entity.get("reaction", "") == "friendly":
                continue
            
            # Calculate distance to target
            distance = 0
            if (hasattr(state, "player_position") and state.player_position and 
                    hasattr(entity, "position") and entity.get("position")):
                px, py, pz = state.player_position
                ex, ey, ez = entity.get("position")
                distance = math.sqrt((px-ex)**2 + (py-ey)**2 + (pz-ez)**2)
            
            # Calculate level difference penalty
            level_diff = entity_level - player_level
            level_penalty = 0
            if level_diff > 5:  # Too high level
                level_penalty = 100  # Very high penalty
            elif level_diff > 3:
                level_penalty = 50
            elif level_diff > 0:
                level_penalty = 10 * level_diff
            
            # Base score is inverse of distance (closer is better)
            # Adjusted by level difference
            score = 100 - min(distance * 5, 80) - level_penalty
            
            # Bonus for quest targets
            is_quest_target = False
            if hasattr(state, "active_quests") and state.active_quests:
                for quest in state.active_quests:
                    if any(objective.get("target", "").lower() in entity_id.lower() 
                          for objective in quest.get("objectives", [])):
                        is_quest_target = True
                        score += 30
                        break
            
            hostile_targets.append({
                "id": entity_id,
                "reason": f"Hostile target {entity_id}" + (" (quest)" if is_quest_target else ""),
                "score": score
            })
        
        # Combine target lists and sort by score
        targets.extend(hostile_targets)
        targets.sort(key=lambda t: t["score"], reverse=True)
        
        if targets:
            return targets[0]
        
        return target_info
    
    def get_optimal_position(self, state: GameState, specialization: str = None) -> Dict:
        """
        Get the optimal position based on the priest's specialization and game state
        
        Args:
            state: Current game state
            specialization: Talent specialization
            
        Returns:
            Position information
        """
        if not specialization:
            specialization = self._detect_specialization(state)
        
        position_info = {
            "type": "position",
            "x": 0,
            "y": 0,
            "z": 0,
            "description": "Default position",
            "reason": "No specific positioning needed"
        }
        
        # Get target information
        target_id = state.target if hasattr(state, "target") and state.target else None
        target_entity = None
        
        if target_id and hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                if entity.get("id") == target_id:
                    target_entity = entity
                    break
        
        # No target, can't calculate position
        if not target_entity:
            return position_info
        
        # Get current positions
        if not (hasattr(state, "player_position") and state.player_position):
            return position_info
        
        player_pos = state.player_position
        target_pos = target_entity.get("position")
        
        if not target_pos:
            return position_info
        
        # Calculate distance to target
        px, py, pz = player_pos
        tx, ty, tz = target_pos
        current_distance = math.sqrt((px-tx)**2 + (py-ty)**2 + (pz-tz)**2)
        
        # Determine optimal range based on specialization
        if specialization == "shadow":
            # Shadow priests want to be at 20-30 yards for maximum casting time
            optimal_distance = 25
            
            if current_distance < 15:
                # Need to move away
                direction_x = px - tx
                direction_y = py - ty
                
                # Normalize direction
                magnitude = math.sqrt(direction_x**2 + direction_y**2)
                if magnitude > 0:
                    direction_x /= magnitude
                    direction_y /= magnitude
                
                # Calculate new position 
                new_x = px + direction_x * 15
                new_y = py + direction_y * 15
                
                position_info = {
                    "type": "position",
                    "x": new_x,
                    "y": new_y,
                    "z": pz,
                    "description": "Move away from target",
                    "reason": "Maintain optimal shadow casting distance"
                }
        else:
            # Healing specs generally want to be with the group
            # In a real implementation, would calculate group center
            if current_distance > 30:
                # Move closer to target if healing
                direction_x = tx - px
                direction_y = ty - py
                
                # Normalize direction
                magnitude = math.sqrt(direction_x**2 + direction_y**2)
                if magnitude > 0:
                    direction_x /= magnitude
                    direction_y /= magnitude
                
                # Calculate new position to get within 20 yards
                move_distance = current_distance - 20
                new_x = px + direction_x * move_distance
                new_y = py + direction_y * move_distance
                
                position_info = {
                    "type": "position",
                    "x": new_x,
                    "y": new_y,
                    "z": pz,
                    "description": "Move closer to target",
                    "reason": "Get within healing range"
                }
        
        return position_info
    
    def get_resource_abilities(self, state: GameState, specialization: str = None) -> List[Dict]:
        """
        Get abilities that help with resource management (mana)
        
        Args:
            state: Current game state
            specialization: Talent specialization
            
        Returns:
            List of resource management abilities
        """
        abilities = []
        player_mana_percent = state.player_mana_percent if hasattr(state, "player_mana_percent") else 100
        
        # Only suggest resource abilities if mana is getting low
        if player_mana_percent < 40:
            # Shadowfiend is available to all priests for mana recovery
            abilities.append({
                "type": "cast",
                "spell": "Shadowfiend",
                "target": state.target if hasattr(state, "target") and state.target else None,
                "description": "Summon Shadowfiend for mana recovery"
            })
            
            # Hymn of Hope for significant mana recovery
            if player_mana_percent < 20:
                abilities.append({
                    "type": "cast",
                    "spell": "Hymn of Hope",
                    "target": "player",
                    "description": "Channel for mana recovery"
                })
            
            # Inner Focus for a free spell (less mana spent)
            abilities.append({
                "type": "cast",
                "spell": "Inner Focus",
                "target": "player",
                "description": "Next heal costs no mana and has 25% increased critical chance" 
            })
            
            # Discipline-specific ability
            if specialization == "discipline":
                abilities.append({
                    "type": "cast",
                    "spell": "Power Infusion",
                    "target": "player", 
                    "description": "20% increased casting speed and reduced mana cost"
                })
        
        return abilities
    
    def get_defensive_abilities(self, state: GameState, specialization: str = None) -> List[Dict]:
        """
        Get defensive abilities based on the current situation
        
        Args:
            state: Current game state
            specialization: Talent specialization
            
        Returns:
            List of defensive abilities
        """
        abilities = []
        player_health_percent = state.player_health_percent if hasattr(state, "player_health_percent") else 100
        
        # Power Word: Shield (if not on cooldown)
        if self.power_word_shield_cooldown <= 0:
            abilities.append({
                "type": "cast",
                "spell": "Power Word: Shield",
                "target": "player",
                "description": "Shield to prevent damage"
            })
        
        # Desperate emergency (very low health)
        if player_health_percent < 30:
            # Desperate Prayer for emergency healing
            abilities.append({
                "type": "cast",
                "spell": "Desperate Prayer",
                "target": "player",
                "description": "Emergency self-heal"
            })
            
            # Holy-specific major cooldown
            if specialization == "holy":
                abilities.append({
                    "type": "cast",
                    "spell": "Guardian Spirit",
                    "target": "player",
                    "description": "Prevent death and increase healing"
                })
            
            # Discipline-specific major cooldown
            if specialization == "discipline":
                abilities.append({
                    "type": "cast",
                    "spell": "Pain Suppression",
                    "target": "player",
                    "description": "Reduce damage taken by 40%"
                })
            
            # Flash Heal for emergency healing
            abilities.append({
                "type": "cast",
                "spell": "Flash Heal",
                "target": "player",
                "description": "Quick emergency heal"
            })
        
        # Fear ability when surrounded
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            close_hostile_count = sum(1 for entity in state.nearby_entities 
                                     if entity.get("reaction") == "hostile" and
                                     self._get_distance(state.player_position, entity.get("position", (0,0,0))) < 8)
            
            if close_hostile_count >= 3:
                abilities.append({
                    "type": "cast",
                    "spell": "Psychic Scream",
                    "target": None,
                    "description": "Fear surrounding enemies"
                })
        
        # Fade to drop threat
        if hasattr(state, "threat_level") and getattr(state, "threat_level", 0) > 80:
            abilities.append({
                "type": "cast",
                "spell": "Fade",
                "target": "player",
                "description": "Temporarily reduce threat"
            })
        
        return abilities
    
    def _get_distance(self, pos1, pos2):
        """Helper method to calculate distance between positions"""
        if not pos1 or not pos2:
            return 999  # Large default distance
        
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    def get_supported_talent_builds(self) -> List[Dict[str, Any]]:
        """
        Get the list of supported talent builds for Priest
        
        Returns:
            List of talent build descriptions
        """
        return [
            {
                "name": "Discipline",
                "description": "Focuses on shields, damage prevention, and efficient healing",
                "key_abilities": ["Power Word: Shield", "Penance", "Pain Suppression"],
                "strengths": ["Mana efficiency", "Damage prevention", "Single-target healing"],
                "weaknesses": ["Limited AoE healing", "Relies on anticipating damage"],
                "difficulty": "Medium"
            },
            {
                "name": "Holy",
                "description": "Versatile healing specialization with strong group healing",
                "key_abilities": ["Circle of Healing", "Guardian Spirit", "Prayer of Healing"],
                "strengths": ["Group healing", "Healing power", "Versatility"],
                "weaknesses": ["Mana intensive", "Limited damage capability"],
                "difficulty": "Medium"
            },
            {
                "name": "Shadow",
                "description": "Dark magic damage dealer focusing on DoTs and Mind spells",
                "key_abilities": ["Shadowform", "Mind Blast", "Shadow Word: Pain", "Mind Flay"],
                "strengths": ["Sustained damage", "Self healing through Vampiric abilities", "Mind Control utility"],
                "weaknesses": ["Less effective healing", "Target switching", "Long DoT ramp-up time"],
                "difficulty": "High"
            }
        ]