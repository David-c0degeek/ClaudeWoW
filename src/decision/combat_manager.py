"""
Combat Manager Module

This module handles combat-related decision making and planning.
"""

import logging
import random
from typing import Dict, List, Tuple, Any, Optional
import time

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge

class CombatManager:
    """
    Manages combat decisions and tactics
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
        
        # Cache for combat rotations
        self.rotation_cache = {}
        self.cache_ttl = 60  # seconds
        self.last_cache_update = 0
        
        # Current combat state
        self.current_target = None
        self.current_spell_index = 0
        self.spell_history = []
        self.last_cast_time = 0
        self.global_cooldown = 1.5  # seconds
        
        self.logger.info("CombatManager initialized")
    
    def generate_combat_plan(self, state: GameState) -> List[Dict]:
        """
        Generate a combat plan based on the current game state
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Combat plan steps
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
            target = state.target if state.target else suitable_target.get("id")
            target_health = state.target_health if hasattr(state, "target_health") else 100
            
            # Get player class and level
            player_class = state.player_class if hasattr(state, "player_class") and state.player_class else "warrior"
            player_level = state.player_level if hasattr(state, "player_level") and state.player_level else 1
            
            # Get the optimal combat rotation
            rotation = self._get_combat_rotation(player_class, player_level, state)
            
            # Adjust for target health percentage
            if target_health < 20:
                # Add execute abilities for low health targets
                execute_abilities = self._get_execute_abilities(player_class)
                if execute_abilities:
                    rotation = execute_abilities + rotation
            
            # Add positioning step if needed
            if self._needs_repositioning(state):
                optimal_position = self._calculate_optimal_position(state)
                plan.append({
                    "type": "move",
                    "position": optimal_position,
                    "description": "Move to optimal combat position"
                })
            
            # Add spell casting steps
            for spell in rotation[:5]:  # Limit to next 5 spells
                plan.append({
                    "type": "cast",
                    "spell": spell.get("name"),
                    "target": target,
                    "description": f"Cast {spell.get('name')} on {target}"
                })
            
            # Add contingency steps
            if player_class in ["priest", "druid", "paladin", "shaman"]:
                # Check if healing is needed
                if state.player_health < 50:
                    heal_spell = self._get_best_healing_spell(player_class, player_level)
                    if heal_spell:
                        plan.append({
                            "type": "cast",
                            "spell": heal_spell.get("name"),
                            "target": "self",
                            "description": f"Cast {heal_spell.get('name')} on self"
                        })
            
            # Add defensive cooldown if low health
            if state.player_health < 30:
                defensive_ability = self._get_defensive_ability(player_class, player_level)
                if defensive_ability:
                    plan.insert(0, {  # Insert at beginning of plan
                        "type": "cast",
                        "spell": defensive_ability.get("name"),
                        "target": "self",
                        "description": f"Cast defensive {defensive_ability.get('name')}"
                    })
        
        except Exception as e:
            self.logger.error(f"Error generating combat plan: {e}")
            # Fallback to basic attack
            plan = [{
                "type": "cast",
                "spell": "Attack",
                "target": state.target if state.target else "target",
                "description": "Basic attack"
            }]
        
        return plan
    
    def _find_suitable_target(self, state: GameState) -> Optional[Dict]:
        """
        Find a suitable combat target
        
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
    
    def _get_combat_rotation(self, player_class: str, player_level: int, state: GameState) -> List[Dict]:
        """
        Get the optimal combat rotation for the player's class and level
        
        Args:
            player_class: Player's class
            player_level: Player's level
            state: Current game state
        
        Returns:
            List[Dict]: List of spells to cast in order
        """
        # Check cache first
        cache_key = f"{player_class}_{player_level}"
        current_time = time.time()
        
        if (cache_key in self.rotation_cache and 
            current_time - self.last_cache_update < self.cache_ttl):
            return self.rotation_cache[cache_key]
        
        # Query knowledge base for rotation
        rotation = self.knowledge.get_combat_rotation(player_class, player_level)
        
        # If rotation not found, use default based on class
        if not rotation:
            rotation = self._get_default_rotation(player_class)
        
        # Update cache
        self.rotation_cache[cache_key] = rotation
        self.last_cache_update = current_time
        
        return rotation
    
    def _get_default_rotation(self, player_class: str) -> List[Dict]:
        """
        Get a default rotation for a class when no specific rotation is available
        
        Args:
            player_class: Player's class
        
        Returns:
            List[Dict]: Default rotation
        """
        default_rotations = {
            "warrior": [
                {"name": "Battle Shout", "type": "buff"},
                {"name": "Charge", "type": "ability"},
                {"name": "Rend", "type": "ability"},
                {"name": "Thunder Clap", "type": "ability"},
                {"name": "Heroic Strike", "type": "ability"}
            ],
            "paladin": [
                {"name": "Seal of Righteousness", "type": "buff"},
                {"name": "Judgement", "type": "ability"},
                {"name": "Crusader Strike", "type": "ability"},
                {"name": "Consecration", "type": "ability"},
                {"name": "Flash of Light", "type": "heal"}
            ],
            "hunter": [
                {"name": "Hunter's Mark", "type": "ability"},
                {"name": "Serpent Sting", "type": "ability"},
                {"name": "Arcane Shot", "type": "ability"},
                {"name": "Steady Shot", "type": "ability"},
                {"name": "Kill Command", "type": "ability"}
            ],
            "rogue": [
                {"name": "Stealth", "type": "buff"},
                {"name": "Cheap Shot", "type": "ability"},
                {"name": "Sinister Strike", "type": "ability"},
                {"name": "Slice and Dice", "type": "ability"},
                {"name": "Eviscerate", "type": "ability"}
            ],
            "priest": [
                {"name": "Power Word: Shield", "type": "buff"},
                {"name": "Shadow Word: Pain", "type": "ability"},
                {"name": "Mind Blast", "type": "ability"},
                {"name": "Smite", "type": "ability"},
                {"name": "Renew", "type": "heal"}
            ],
            "shaman": [
                {"name": "Lightning Shield", "type": "buff"},
                {"name": "Flame Shock", "type": "ability"},
                {"name": "Lightning Bolt", "type": "ability"},
                {"name": "Earth Shock", "type": "ability"},
                {"name": "Healing Wave", "type": "heal"}
            ],
            "mage": [
                {"name": "Arcane Intellect", "type": "buff"},
                {"name": "Frostbolt", "type": "ability"},
                {"name": "Fire Blast", "type": "ability"},
                {"name": "Arcane Missiles", "type": "ability"},
                {"name": "Frost Nova", "type": "ability"}
            ],
            "warlock": [
                {"name": "Summon Imp", "type": "buff"},
                {"name": "Corruption", "type": "ability"},
                {"name": "Shadow Bolt", "type": "ability"},
                {"name": "Immolate", "type": "ability"},
                {"name": "Life Tap", "type": "ability"}
            ],
            "druid": [
                {"name": "Mark of the Wild", "type": "buff"},
                {"name": "Moonfire", "type": "ability"},
                {"name": "Wrath", "type": "ability"},
                {"name": "Rejuvenation", "type": "heal"},
                {"name": "Entangling Roots", "type": "ability"}
            ]
        }
        
        # Default to warrior if class not found
        return default_rotations.get(player_class.lower(), default_rotations["warrior"])
    
    def _get_execute_abilities(self, player_class: str) -> List[Dict]:
        """
        Get execute-phase abilities for a given class
        
        Args:
            player_class: Player's class
        
        Returns:
            List[Dict]: Execute abilities
        """
        execute_abilities = {
            "warrior": [{"name": "Execute", "type": "ability"}],
            "hunter": [{"name": "Kill Shot", "type": "ability"}],
            "mage": [{"name": "Fire Blast", "type": "ability"}],
            "warlock": [{"name": "Shadowburn", "type": "ability"}],
            "paladin": [{"name": "Hammer of Wrath", "type": "ability"}],
            "priest": [{"name": "Shadow Word: Death", "type": "ability"}],
            "death knight": [{"name": "Soul Reaper", "type": "ability"}]
        }
        
        return execute_abilities.get(player_class.lower(), [])
    
    def _get_best_healing_spell(self, player_class: str, player_level: int) -> Optional[Dict]:
        """
        Get the best healing spell available for the player's class and level
        
        Args:
            player_class: Player's class
            player_level: Player's level
        
        Returns:
            Optional[Dict]: Best healing spell or None
        """
        healing_spells = {
            "priest": [
                {"name": "Flash Heal", "min_level": 20, "type": "heal"},
                {"name": "Renew", "min_level": 8, "type": "heal"},
                {"name": "Lesser Heal", "min_level": 1, "type": "heal"}
            ],
            "druid": [
                {"name": "Healing Touch", "min_level": 1, "type": "heal"},
                {"name": "Rejuvenation", "min_level": 4, "type": "heal"},
                {"name": "Regrowth", "min_level": 12, "type": "heal"}
            ],
            "paladin": [
                {"name": "Flash of Light", "min_level": 20, "type": "heal"},
                {"name": "Holy Light", "min_level": 1, "type": "heal"}
            ],
            "shaman": [
                {"name": "Healing Wave", "min_level": 1, "type": "heal"},
                {"name": "Lesser Healing Wave", "min_level": 20, "type": "heal"},
                {"name": "Chain Heal", "min_level": 40, "type": "heal"}
            ]
        }
        
        # Check if class has healing spells
        if player_class.lower() not in healing_spells:
            return None
        
        # Find best healing spell based on level
        available_spells = [
            spell for spell in healing_spells[player_class.lower()]
            if spell["min_level"] <= player_level
        ]
        
        if available_spells:
            # Return the highest level spell
            return max(available_spells, key=lambda s: s["min_level"])
        
        return None
    
    def _get_defensive_ability(self, player_class: str, player_level: int) -> Optional[Dict]:
        """
        Get the best defensive ability available for the player's class and level
        
        Args:
            player_class: Player's class
            player_level: Player's level
        
        Returns:
            Optional[Dict]: Defensive ability or None
        """
        defensive_abilities = {
            "warrior": [
                {"name": "Shield Wall", "min_level": 28, "type": "defensive"},
                {"name": "Last Stand", "min_level": 16, "type": "defensive"}
            ],
            "paladin": [
                {"name": "Divine Shield", "min_level": 18, "type": "defensive"},
                {"name": "Lay on Hands", "min_level": 10, "type": "defensive"}
            ],
            "hunter": [
                {"name": "Feign Death", "min_level": 30, "type": "defensive"},
                {"name": "Deterrence", "min_level": 20, "type": "defensive"}
            ],
            "rogue": [
                {"name": "Evasion", "min_level": 8, "type": "defensive"},
                {"name": "Vanish", "min_level": 22, "type": "defensive"}
            ],
            "priest": [
                {"name": "Power Word: Shield", "min_level": 6, "type": "defensive"},
                {"name": "Fade", "min_level": 8, "type": "defensive"}
            ],
            "shaman": [
                {"name": "Earth Shield", "min_level": 50, "type": "defensive"},
                {"name": "Shamanistic Rage", "min_level": 40, "type": "defensive"}
            ],
            "mage": [
                {"name": "Ice Block", "min_level": 30, "type": "defensive"},
                {"name": "Frost Nova", "min_level": 10, "type": "defensive"}
            ],
            "warlock": [
                {"name": "Soulstone", "min_level": 18, "type": "defensive"},
                {"name": "Unending Resolve", "min_level": 42, "type": "defensive"}
            ],
            "druid": [
                {"name": "Barkskin", "min_level": 44, "type": "defensive"},
                {"name": "Survival Instincts", "min_level": 20, "type": "defensive"}
            ]
        }
        
        # Check if class has defensive abilities
        if player_class.lower() not in defensive_abilities:
            return None
        
        # Find best defensive ability based on level
        available_abilities = [
            ability for ability in defensive_abilities[player_class.lower()]
            if ability["min_level"] <= player_level
        ]
        
        if available_abilities:
            # Return the highest level ability
            return max(available_abilities, key=lambda a: a["min_level"])
        
        return None
    
    def _needs_repositioning(self, state: GameState) -> bool:
        """
        Determine if the player needs to reposition during combat
        
        Args:
            state: Current game state
        
        Returns:
            bool: True if repositioning is needed
        """
        # Simple implementation - check if target exists but no target position
        if state.target and not hasattr(state, "target_position"):
            return True
        
        # Check if player is too close or too far from target
        if (hasattr(state, "player_position") and 
            hasattr(state, "target_position") and 
            state.player_position and 
            state.target_position):
            
            # Calculate distance to target
            dx = state.player_position[0] - state.target_position[0]
            dy = state.player_position[1] - state.target_position[1]
            distance = (dx * dx + dy * dy) ** 0.5
            
            # Get player class to determine optimal range
            player_class = state.player_class if hasattr(state, "player_class") else "warrior"
            
            # Define optimal ranges for different classes
            optimal_ranges = {
                "warrior": (0, 5),     # Melee
                "paladin": (0, 5),     # Melee
                "rogue": (0, 5),       # Melee
                "hunter": (8, 30),     # Ranged
                "mage": (15, 30),      # Ranged
                "warlock": (15, 30),   # Ranged
                "priest": (15, 30),    # Ranged
                "druid": (0, 30),      # Both, depends on form
                "shaman": (0, 30)      # Both, depends on spec
            }
            
            min_range, max_range = optimal_ranges.get(player_class.lower(), (0, 5))
            
            # Check if outside optimal range
            return distance < min_range or distance > max_range
        
        return False
    
    def _calculate_optimal_position(self, state: GameState) -> Tuple[float, float]:
        """
        Calculate the optimal position for combat
        
        Args:
            state: Current game state
        
        Returns:
            Tuple[float, float]: Optimal position coordinates
        """
        if not hasattr(state, "target_position") or not state.target_position:
            # If no target position, return current position
            return state.player_position if hasattr(state, "player_position") else (0, 0)
        
        # Get player class to determine optimal range
        player_class = state.player_class if hasattr(state, "player_class") else "warrior"
        
        # Define optimal distances for different classes
        optimal_distances = {
            "warrior": 3,     # Melee
            "paladin": 3,     # Melee
            "rogue": 3,       # Melee
            "hunter": 25,     # Ranged
            "mage": 25,       # Ranged
            "warlock": 25,    # Ranged
            "priest": 25,     # Ranged
            "druid": 3,       # Default to melee
            "shaman": 20     # Default to medium range
        }
        
        optimal_distance = optimal_distances.get(player_class.lower(), 3)
        
        # Calculate direction vector from target to player
        target_x, target_y = state.target_position
        
        # Generate a random angle to avoid always being in the same spot
        angle = random.uniform(0, 2 * 3.14159)
        
        # Calculate optimal position
        optimal_x = target_x + optimal_distance * math.cos(angle)
        optimal_y = target_y + optimal_distance * math.sin(angle)
        
        return (optimal_x, optimal_y)