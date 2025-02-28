"""
Combat Situational Awareness Module

This module analyzes the combat situation and provides tactical awareness 
for making better combat decisions, including:
- AOE detection and opportunities
- Interrupt priorities
- Crowd control management
- PvP vs PvE strategy switching
- Group role awareness and coordination
"""

import logging
import math
import time
from typing import Dict, List, Set, Tuple, Any, Optional

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge


class CombatSituationalAwareness:
    """
    Provides combat situation analysis and tactical awareness across different scenarios
    
    This class analyzes the current game state to provide combat insights such as:
    - When to use AOE abilities vs single target
    - What spells to interrupt and their priority
    - When to use crowd control
    - How to adapt strategies between PvP and PvE
    - How to coordinate in group settings based on role
    """
    
    def __init__(self, config: Dict[str, Any], knowledge: GameKnowledge):
        """
        Initialize the combat situational awareness system
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge database
        """
        self.logger = logging.getLogger("wow_ai.decision.combat.situational_awareness")
        self.config = config
        self.knowledge = knowledge
        
        # AOE tracking
        self.enemy_clusters: List[Dict[str, Any]] = []
        self.aoe_opportunity: bool = False
        self.aoe_enemy_count: int = 0
        self.aoe_threshold: int = 3  # Default threshold for AOE abilities
        
        # Interrupt tracking
        self.interrupt_targets: List[Dict[str, Any]] = []  # Entities casting interruptible spells
        self.cc_targets: List[Dict[str, Any]] = []  # Entities that should be CC'd
        
        # PvP detection
        self.in_pvp: bool = False
        self.pvp_enemies: List[Dict[str, Any]] = []
        
        # Group role tracking
        self.is_in_group: bool = False
        self.group_role: str = "dps"  # "tank", "healer", "dps"
        self.group_members: List[Dict[str, Any]] = []
        self.tank_target: Optional[str] = None
        
        # Dangerous ability warnings
        self.active_enemy_abilities: List[Dict[str, Any]] = []  # Abilities being cast or active that are dangerous
        self.danger_level: int = 0  # 0-10 scale of current combat danger level
        
        # Tactical suggestions
        self.tactical_suggestions: List[str] = []
        
        self.logger.info("Combat situational awareness initialized")
    
    def update(self, state: GameState) -> None:
        """
        Update situational awareness based on current game state
        
        Args:
            state: Current game state
        """
        # Clear previous state
        self.enemy_clusters = []
        self.interrupt_targets = []
        self.cc_targets = []
        self.pvp_enemies = []
        self.active_enemy_abilities = []
        self.tactical_suggestions = []
        
        # Update based on new state
        self._analyze_enemy_clusters(state)
        self._identify_interrupt_targets(state)
        self._detect_pvp_situation(state)
        self._analyze_group_situation(state)
        self._identify_dangerous_abilities(state)
        self._generate_tactical_suggestions(state)
    
    def _analyze_enemy_clusters(self, state: GameState) -> None:
        """
        Analyze enemy positions to identify clusters for AOE abilities
        
        Args:
            state: Current game state
        """
        if not hasattr(state, "nearby_entities") or not state.nearby_entities:
            self.aoe_opportunity = False
            self.aoe_enemy_count = 0
            return
        
        # Extract enemy positions
        enemy_positions = []
        for entity in state.nearby_entities:
            if (entity.get("reaction") == "hostile" and 
                entity.get("type") == "mob" and
                entity.get("position") and
                entity.get("health_percent", 0) > 0):
                
                enemy_positions.append({
                    "id": entity.get("id"),
                    "position": entity.get("position"),
                    "health": entity.get("health_percent", 100),
                    "level": entity.get("level", 1)
                })
        
        # No enemies found
        if not enemy_positions:
            self.aoe_opportunity = False
            self.aoe_enemy_count = 0
            return
        
        # Find clusters using a simple distance-based algorithm
        clusters = []
        remaining = set(range(len(enemy_positions)))
        
        while remaining:
            cluster = []
            seed_idx = next(iter(remaining))
            seed_pos = enemy_positions[seed_idx]["position"]
            remaining.remove(seed_idx)
            cluster.append(enemy_positions[seed_idx])
            
            # Find all nearby enemies
            for i in list(remaining):
                curr_pos = enemy_positions[i]["position"]
                distance = math.sqrt((seed_pos[0] - curr_pos[0])**2 + (seed_pos[1] - curr_pos[1])**2)
                
                if distance <= 8:  # 8 yards is a common AOE radius
                    cluster.append(enemy_positions[i])
                    remaining.remove(i)
            
            if len(cluster) >= 2:  # Only consider as cluster if at least 2 enemies
                center_x = sum(e["position"][0] for e in cluster) / len(cluster)
                center_y = sum(e["position"][1] for e in cluster) / len(cluster)
                
                clusters.append({
                    "center": (center_x, center_y),
                    "size": len(cluster),
                    "entities": cluster
                })
        
        # Sort clusters by size
        clusters.sort(key=lambda c: c["size"], reverse=True)
        self.enemy_clusters = clusters
        
        # Determine if AOE is worthwhile
        self.aoe_enemy_count = max([c["size"] for c in clusters]) if clusters else 0
        self.aoe_opportunity = self.aoe_enemy_count >= self.aoe_threshold
        
        if self.aoe_opportunity:
            self.logger.debug(f"AOE opportunity detected with {self.aoe_enemy_count} enemies")
    
    def _identify_interrupt_targets(self, state: GameState) -> None:
        """
        Identify enemies casting spells that should be interrupted
        
        Args:
            state: Current game state
        """
        if not hasattr(state, "nearby_entities") or not state.nearby_entities:
            return
        
        interrupt_targets = []
        
        for entity in state.nearby_entities:
            if entity.get("reaction") != "hostile":
                continue
            
            # Check if entity is casting
            if entity.get("casting"):
                cast_info = entity.get("casting")
                spell_name = cast_info.get("spell")
                
                # Check if spell should be interrupted
                priority = self.knowledge.get_spell_interrupt_priority(spell_name)
                
                if priority > 0:
                    interrupt_targets.append({
                        "id": entity.get("id"),
                        "spell": spell_name,
                        "priority": priority,
                        "remaining": cast_info.get("remaining_time", 1.5),
                        "position": entity.get("position")
                    })
                    
                    # Check for CC opportunity
                    if self.knowledge.is_spell_cc_priority(spell_name) and entity.get("id") not in [t["id"] for t in self.cc_targets]:
                        self.cc_targets.append({
                            "id": entity.get("id"),
                            "reason": f"Casting {spell_name}",
                            "priority": priority
                        })
        
        # Sort by priority and remaining cast time
        interrupt_targets.sort(key=lambda t: (t["priority"], -t["remaining"]), reverse=True)
        self.interrupt_targets = interrupt_targets
        
        if interrupt_targets:
            self.logger.debug(f"Identified {len(interrupt_targets)} interrupt targets")
    
    def _detect_pvp_situation(self, state: GameState) -> None:
        """
        Detect if we're in a PvP situation and identify enemy players
        
        Args:
            state: Current game state
        """
        if not hasattr(state, "nearby_entities") or not state.nearby_entities:
            self.in_pvp = False
            return
        
        # Look for hostile players
        pvp_enemies = []
        for entity in state.nearby_entities:
            if entity.get("type") == "player" and entity.get("reaction") == "hostile":
                pvp_enemies.append({
                    "id": entity.get("id"),
                    "class": entity.get("class", "unknown"),
                    "level": entity.get("level", 60),
                    "health": entity.get("health_percent", 100),
                    "position": entity.get("position"),
                    "target": entity.get("target")
                })
        
        self.pvp_enemies = pvp_enemies
        self.in_pvp = len(pvp_enemies) > 0
        
        if self.in_pvp:
            self.logger.debug(f"PvP situation detected with {len(pvp_enemies)} enemy players")
    
    def _analyze_group_situation(self, state: GameState) -> None:
        """
        Analyze group composition and roles
        
        Args:
            state: Current game state
        """
        # Check if we're in a group
        if hasattr(state, "group_members") and state.group_members:
            self.is_in_group = True
            self.group_members = []
            
            # Process group members
            for member_id in state.group_members:
                member_info = {
                    "id": member_id,
                    "health": state.group_members_health.get(member_id, 100) if hasattr(state, "group_members_health") else 100,
                    "role": "unknown"
                }
                
                # Try to determine role
                if hasattr(state, f"{member_id}_class"):
                    member_class = getattr(state, f"{member_id}_class")
                    member_info["class"] = member_class
                    
                    # Deduce role from class and markers
                    if member_class in ["Warrior", "Paladin", "Druid", "Death Knight"]:
                        if hasattr(state, f"{member_id}_role") and getattr(state, f"{member_id}_role") == "tank":
                            member_info["role"] = "tank"
                            # Track tank's target
                            if hasattr(state, f"{member_id}_target"):
                                self.tank_target = getattr(state, f"{member_id}_target")
                    
                    if member_class in ["Priest", "Paladin", "Druid", "Shaman"]:
                        if hasattr(state, f"{member_id}_role") and getattr(state, f"{member_id}_role") == "healer":
                            member_info["role"] = "healer"
                
                self.group_members.append(member_info)
            
            # Determine our own role
            if hasattr(state, "player_role"):
                self.group_role = state.player_role
            else:
                # Try to deduce from class and spec
                player_class = state.player_class if hasattr(state, "player_class") else "unknown"
                
                if player_class == "Warrior":
                    # Check for defensive stance
                    if any(buff == "Defensive Stance" for buff in state.player_buffs) if hasattr(state, "player_buffs") else False:
                        self.group_role = "tank"
                elif player_class == "Priest":
                    # Check for shadowform
                    if any(buff == "Shadowform" for buff in state.player_buffs) if hasattr(state, "player_buffs") else False:
                        self.group_role = "dps"
                    else:
                        self.group_role = "healer"
        else:
            self.is_in_group = False
            self.group_members = []
            self.tank_target = None
    
    def _identify_dangerous_abilities(self, state: GameState) -> None:
        """
        Identify dangerous enemy abilities being cast or active
        
        Args:
            state: Current game state
        """
        dangerous_abilities = []
        danger_level = 0
        
        # Check for enemy casts
        if hasattr(state, "nearby_entities") and state.nearby_entities:
            for entity in state.nearby_entities:
                if entity.get("casting"):
                    cast_info = entity.get("casting")
                    spell_name = cast_info.get("spell")
                    
                    # Check if spell is dangerous
                    danger_rating = self.knowledge.get_spell_danger_rating(spell_name)
                    
                    if danger_rating > 3:  # On a scale of 1-10
                        dangerous_abilities.append({
                            "name": spell_name,
                            "caster": entity.get("id"),
                            "danger_rating": danger_rating,
                            "remaining": cast_info.get("remaining_time", 1.5),
                            "type": "cast"
                        })
                        danger_level = max(danger_level, danger_rating)
        
        # Check for ground effects or boss mechanics
        if hasattr(state, "ground_effects") and state.ground_effects:
            for effect in state.ground_effects:
                effect_name = effect.get("name", "")
                danger_rating = self.knowledge.get_ground_effect_danger(effect_name)
                
                if danger_rating > 3:
                    dangerous_abilities.append({
                        "name": effect_name,
                        "position": effect.get("position"),
                        "radius": effect.get("radius", 5),
                        "danger_rating": danger_rating,
                        "remaining": effect.get("remaining_time", 5),
                        "type": "ground_effect"
                    })
                    danger_level = max(danger_level, danger_rating)
        
        self.active_enemy_abilities = dangerous_abilities
        self.danger_level = danger_level
        
        if dangerous_abilities:
            self.logger.debug(f"Identified {len(dangerous_abilities)} dangerous abilities, danger level: {danger_level}/10")
    
    def _generate_tactical_suggestions(self, state: GameState) -> None:
        """
        Generate tactical suggestions based on all analyses
        
        Args:
            state: Current game state
        """
        suggestions = []
        
        # AOE suggestions
        if self.aoe_opportunity:
            suggestions.append(f"Use AOE abilities on cluster of {self.aoe_enemy_count} enemies")
        
        # Interrupt suggestions
        if self.interrupt_targets:
            top_target = self.interrupt_targets[0]
            suggestions.append(f"Interrupt {top_target['spell']} being cast by {top_target['id']}")
        
        # CC suggestions
        if self.cc_targets:
            top_cc = self.cc_targets[0]
            suggestions.append(f"Crowd control {top_cc['id']} ({top_cc['reason']})")
        
        # PvP suggestions
        if self.in_pvp:
            # Identify key targets like healers
            healers = [e for e in self.pvp_enemies if e.get("class") in ["Priest", "Druid", "Paladin", "Shaman"]]
            if healers:
                suggestions.append(f"Focus PvP healer: {healers[0]['id']} ({healers[0]['class']})")
            
            # Suggest defensive play if outnumbered
            if len(self.pvp_enemies) > 2:
                suggestions.append("Play defensively in PvP - outnumbered")
        
        # Group role suggestions
        if self.is_in_group:
            if self.group_role == "tank":
                # Get threat suggestions
                suggestions.append("Maintain threat on all targets")
                
                # Position suggestions
                if self.active_enemy_abilities:
                    suggestions.append("Position boss away from group")
            
            elif self.group_role == "healer":
                # Find who needs healing
                low_health_members = [m for m in self.group_members if m["health"] < 70]
                if low_health_members:
                    suggestions.append(f"Heal {low_health_members[0]['id']} at {low_health_members[0]['health']}% health")
                
                # Position suggestions
                suggestions.append("Stay in range of group but away from AOE")
            
            elif self.group_role == "dps":
                # Target suggestions
                if self.tank_target:
                    suggestions.append(f"Focus tank's target: {self.tank_target}")
        
        # Danger avoidance
        if self.danger_level >= 7:
            suggestions.append(f"DANGER! Use defensive abilities immediately")
        elif self.danger_level >= 4:
            suggestions.append(f"Use defensive cooldown for incoming damage")
        
        self.tactical_suggestions = suggestions
    
    def get_aoe_opportunities(self) -> List[Dict[str, Any]]:
        """
        Get information about AOE opportunities
        
        Returns:
            List[Dict]: List of enemy clusters for AOE
        """
        return self.enemy_clusters
    
    def get_interrupt_targets(self) -> List[Dict[str, Any]]:
        """
        Get targets that should be interrupted
        
        Returns:
            List[Dict]: List of interrupt targets in priority order
        """
        return self.interrupt_targets
    
    def get_cc_targets(self) -> List[Dict[str, Any]]:
        """
        Get targets that should be crowd controlled
        
        Returns:
            List[Dict]: List of CC targets in priority order
        """
        return self.cc_targets
    
    def get_pvp_targets(self) -> List[Dict[str, Any]]:
        """
        Get enemy players in PvP situation
        
        Returns:
            List[Dict]: List of enemy players
        """
        return self.pvp_enemies
    
    def is_pvp_situation(self) -> bool:
        """
        Check if we're in a PvP situation
        
        Returns:
            bool: True if PvP situation is detected
        """
        return self.in_pvp
    
    def get_group_info(self) -> Dict[str, Any]:
        """
        Get information about the current group
        
        Returns:
            Dict: Information about group composition and roles
        """
        return {
            "is_in_group": self.is_in_group,
            "player_role": self.group_role,
            "members": self.group_members,
            "tank_target": self.tank_target
        }
    
    def get_danger_assessment(self) -> Dict[str, Any]:
        """
        Get assessment of current combat danger
        
        Returns:
            Dict: Danger assessment with level and active dangerous abilities
        """
        return {
            "danger_level": self.danger_level,
            "dangerous_abilities": self.active_enemy_abilities
        }
    
    def get_tactical_suggestions(self) -> List[str]:
        """
        Get tactical suggestions for the current combat situation
        
        Returns:
            List[str]: List of tactical suggestions
        """
        return self.tactical_suggestions