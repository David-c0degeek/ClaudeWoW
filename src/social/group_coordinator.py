# src/social/group_coordinator.py

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import random

class GroupCoordinator:
    """
    Manages group coordination and party/raid dynamics
    """
    
    def __init__(self, config: Dict, knowledge_base):
        """
        Initialize the GroupCoordinator
        
        Args:
            config: Configuration dictionary
            knowledge_base: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.social.group_coordinator")
        self.config = config
        self.knowledge_base = knowledge_base
        
        # Group state
        self.in_group = False
        self.in_raid = False
        self.is_leader = False
        self.group_members = []
        self.group_roles = {}  # player_name -> role
        
        # Instance/dungeon state
        self.current_instance = None
        self.instance_knowledge = {}
        
        # Role assignments
        self.player_class = config.get("player_class", "warrior").lower()
        self.player_role = self._determine_player_role()
        
        # Coordination strategies
        self.strategies = {
            "tank": self._tank_strategy,
            "healer": self._healer_strategy,
            "dps": self._dps_strategy,
            "leader": self._leader_strategy
        }
        
        self.logger.info(f"GroupCoordinator initialized with role: {self.player_role}")
    
    def update_group_state(self, state: Any) -> None:
        """
        Update group state from game state
        
        Args:
            state: Current game state
        """
        self.in_group = state.is_in_group if hasattr(state, "is_in_group") else False
        self.in_raid = state.is_in_raid if hasattr(state, "is_in_raid") else False
        
        # Update group members
        if hasattr(state, "group_members"):
            self.group_members = state.group_members
            
            # Analyze group composition
            self._analyze_group_composition()
        
        # Check if player is leader
        if hasattr(state, "is_group_leader"):
            self.is_leader = state.is_group_leader
        
        # Update current instance
        if hasattr(state, "current_instance"):
            if state.current_instance != self.current_instance:
                self.current_instance = state.current_instance
                self._load_instance_knowledge()
    
    def generate_coordination_actions(self, state: Any) -> List[Dict]:
        """
        Generate coordination actions based on group state
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Coordination actions
        """
        if not self.in_group:
            return []
        
        actions = []
        
        # Execute role-specific strategy
        if self.is_leader:
            leader_actions = self.strategies["leader"](state)
            actions.extend(leader_actions)
        
        role_actions = self.strategies[self.player_role](state)
        actions.extend(role_actions)
        
        return actions
    
    def _determine_player_role(self) -> str:
        """
        Determine the player's role based on class and spec
        
        Returns:
            str: Player role (tank, healer, dps)
        """
        player_class = self.player_class
        player_spec = self.config.get("player_spec", "").lower()
        
        # Define role mappings
        tank_specs = {
            "warrior": ["protection"],
            "paladin": ["protection"],
            "druid": ["guardian"],
            "death knight": ["blood"],
            "monk": ["brewmaster"],
            "demon hunter": ["vengeance"]
        }
        
        healer_specs = {
            "priest": ["holy", "discipline"],
            "paladin": ["holy"],
            "druid": ["restoration"],
            "shaman": ["restoration"],
            "monk": ["mistweaver"]
        }
        
        # Check if the player's class and spec match a tank role
        if player_class in tank_specs and player_spec in tank_specs[player_class]:
            return "tank"
        
        # Check if the player's class and spec match a healer role
        if player_class in healer_specs and player_spec in healer_specs[player_class]:
            return "healer"
        
        # Default to DPS
        return "dps"
    
    def _analyze_group_composition(self) -> None:
        """
        Analyze group composition and assign roles
        """
        self.group_roles = {}
        
        for member in self.group_members:
            member_class = member.get("class", "").lower()
            member_spec = member.get("spec", "").lower()
            
            # Determine role based on class and spec
            if member_class in ["warrior", "paladin", "druid", "death knight", "monk", "demon hunter"]:
                if member_spec in ["protection", "guardian", "blood", "brewmaster", "vengeance"]:
                    role = "tank"
                elif member_spec in ["holy", "restoration", "mistweaver", "discipline"]:
                    role = "healer"
                else:
                    role = "dps"
            elif member_class in ["priest", "shaman"]:
                if member_spec in ["holy", "discipline", "restoration"]:
                    role = "healer"
                else:
                    role = "dps"
            else:
                role = "dps"
            
            self.group_roles[member.get("name", "")] = role
    
    def _load_instance_knowledge(self) -> None:
        """
        Load knowledge about the current instance
        """
        if not self.current_instance:
            self.instance_knowledge = {}
            return
        
        # Try to get instance information from knowledge base
        instance_info = self.knowledge_base.get_instance_info(self.current_instance)
        
        if instance_info:
            self.instance_knowledge = instance_info
            self.logger.info(f"Loaded knowledge for instance: {self.current_instance}")
        else:
            self.instance_knowledge = {}
            self.logger.warning(f"No knowledge found for instance: {self.current_instance}")
    
    def _tank_strategy(self, state: Any) -> List[Dict]:
        """
        Generate actions for tank role
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Tank actions
        """
        actions = []
        
        # Check if in combat
        if state.is_in_combat:
            # Tank-specific combat behaviors
            
            # Check if we need to taunt
            if self._need_to_taunt(state):
                actions.append({
                    "type": "cast",
                    "spell": self._get_taunt_ability(),
                    "target": self._get_taunt_target(state),
                    "description": "Taunt threat target"
                })
            
            # Check if we need to use defensive cooldowns
            if state.player_health < 40:
                defensive_ability = self._get_defensive_ability()
                if defensive_ability:
                    actions.append({
                        "type": "cast",
                        "spell": defensive_ability,
                        "target": "self",
                        "description": "Use defensive cooldown"
                    })
        else:
            # Out of combat behaviors
            
            # Check if we should mark targets
            if self.is_leader and self.current_instance:
                marking_actions = self._generate_marking_actions(state)
                actions.extend(marking_actions)
            
            # Check if we should be the first to engage
            if self._should_engage(state):
                target = self._get_next_pull_target(state)
                if target:
                    actions.append({
                        "type": "chat",
                        "message": "Pulling in 3...",
                        "channel": "party",
                        "description": "Announce pull"
                    })
                    
                    # Wait a moment
                    actions.append({
                        "type": "wait",
                        "duration": 3.0,
                        "description": "Wait before pull"
                    })
                    
                    # Pull with range ability if available
                    pull_ability = self._get_pulling_ability()
                    actions.append({
                        "type": "target",
                        "target": target.get("id"),
                        "description": f"Target {target.get('name', 'mob')}"
                    })
                    
                    actions.append({
                        "type": "cast",
                        "spell": pull_ability,
                        "target": target.get("id"),
                        "description": f"Pull {target.get('name', 'mob')}"
                    })
        
        return actions
    
    def _healer_strategy(self, state: Any) -> List[Dict]:
        """
        Generate actions for healer role
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Healer actions
        """
        actions = []
        
        # Check if in combat
        if state.is_in_combat:
            # Healer-specific combat behaviors
            
            # Check if anyone needs healing
            healing_target = self._get_healing_priority(state)
            if healing_target:
                healing_spell = self._get_appropriate_heal(healing_target)
                
                actions.append({
                    "type": "target",
                    "target": healing_target.get("name"),
                    "description": f"Target {healing_target.get('name')} for healing"
                })
                
                actions.append({
                    "type": "cast",
                    "spell": healing_spell,
                    "target": healing_target.get("name"),
                    "description": f"Heal {healing_target.get('name')}"
                })
            
            # Check if we should use group healing cooldown
            if self._need_group_healing(state):
                group_heal = self._get_group_healing_ability()
                actions.append({
                    "type": "cast",
                    "spell": group_heal,
                    "target": "none",
                    "description": "Use group healing ability"
                })
        else:
            # Out of combat behaviors
            
            # Check if we should buff the group
            if self._should_buff_group(state):
                buff_spell = self._get_group_buff()
                actions.append({
                    "type": "cast",
                    "spell": buff_spell,
                    "target": "none",
                    "description": "Buff the group"
                })
            
            # Make sure we have enough mana before next fight
            if state.player_mana < 80 and not self._group_is_moving(state):
                actions.append({
                    "type": "chat",
                    "message": "Need to drink for mana",
                    "channel": "party",
                    "description": "Announce drinking"
                })
                
                actions.append({
                    "type": "use_item",
                    "item": "water",
                    "description": "Drink to restore mana"
                })
        
        return actions
    
    def _dps_strategy(self, state: Any) -> List[Dict]:
        """
        Generate actions for DPS role
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: DPS actions
        """
        actions = []
        
        # Check if in combat
        if state.is_in_combat:
            # DPS-specific combat behaviors
            
            # Check crowd control responsibilities
            cc_target = self._get_cc_target(state)
            if cc_target and self._should_apply_cc(cc_target, state):
                cc_ability = self._get_cc_ability()
                
                actions.append({
                    "type": "target",
                    "target": cc_target.get("id"),
                    "description": f"Target {cc_target.get('name', 'mob')} for CC"
                })
                
                actions.append({
                    "type": "cast",
                    "spell": cc_ability,
                    "target": cc_target.get("id"),
                    "description": f"Apply CC to {cc_target.get('name', 'mob')}"
                })
            
            # Check if we need to use utility abilities
            if self._need_utility(state):
                utility_ability = self._get_utility_ability()
                actions.append({
                    "type": "cast",
                    "spell": utility_ability,
                    "target": "appropriate",
                    "description": "Use utility ability"
                })
            
            # Check if we should focus fire
            if self.group_roles.get("leader") == "tank":
                # Target tank's target
                tank_name = next((name for name, role in self.group_roles.items() if role == "tank"), None)
                if tank_name:
                    tank_target = self._get_player_target(tank_name, state)
                    if tank_target:
                        actions.append({
                            "type": "target",
                            "target": tank_target,
                            "description": "Target same as tank"
                        })
        else:
            # Out of combat behaviors
            
            # Check if we should use group buffs
            if self._should_buff_group(state):
                buff_spell = self._get_group_buff()
                if buff_spell:
                    actions.append({
                        "type": "cast",
                        "spell": buff_spell,
                        "target": "none",
                        "description": "Buff the group"
                    })
        
        return actions
    
    def _leader_strategy(self, state: Any) -> List[Dict]:
        """
        Generate actions for group leader role
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Leader actions
        """
        actions = []
        
        # Check if we should provide instructions
        if self.current_instance and not state.is_in_combat:
            # Check if approaching boss
            approaching_boss = self._approaching_boss(state)
            if approaching_boss:
                boss_info = self._get_boss_info(approaching_boss)
                if boss_info:
                    # Share boss strategy
                    actions.append({
                        "type": "chat",
                        "message": f"Coming up: {approaching_boss}. {boss_info.get('strategy', '')}",
                        "channel": "party",
                        "description": "Share boss strategy"
                    })
            
            # Check for first time party members
            new_members = self._get_inexperienced_members(state)
            if new_members and random.random() < 0.5:  # Only do this sometimes to avoid spam
                instance_tip = self._get_instance_tip()
                actions.append({
                    "type": "chat",
                    "message": f"Tip for this dungeon: {instance_tip}",
                    "channel": "party",
                    "description": "Share instance tip"
                })
        
        # Check if we need to assign roles
        if self._need_role_assignment(state):
            role_message = self._generate_role_assignments()
            actions.append({
                "type": "chat",
                "message": role_message,
                "channel": "party",
                "description": "Assign roles"
            })
        
        return actions
    
    # Helper methods (these would be implemented in a real system)
    def _need_to_taunt(self, state):
        return False  # Placeholder
    
    def _get_taunt_ability(self):
        return "Taunt"  # Placeholder
    
    def _get_taunt_target(self, state):
        return None  # Placeholder
    
    def _get_defensive_ability(self):
        return "Shield Wall"  # Placeholder
    
    def _generate_marking_actions(self, state):
        return []  # Placeholder
    
    def _should_engage(self, state):
        return False  # Placeholder
    
    def _get_next_pull_target(self, state):
        return None  # Placeholder
    
    def _get_pulling_ability(self):
        return "Throw"  # Placeholder
    
    def _get_healing_priority(self, state):
        return None  # Placeholder
    
    def _get_appropriate_heal(self, target):
        return "Heal"  # Placeholder
    
    def _need_group_healing(self, state):
        return False  # Placeholder
    
    def _get_group_healing_ability(self):
        return "Prayer of Healing"  # Placeholder
    
    def _should_buff_group(self, state):
        return False  # Placeholder
    
    def _get_group_buff(self):
        return None  # Placeholder
    
    def _group_is_moving(self, state):
        return True  # Placeholder
    
    def _get_cc_target(self, state):
        return None  # Placeholder
    
    def _should_apply_cc(self, target, state):
        return False  # Placeholder
    
    def _get_cc_ability(self):
        return "Polymorph"  # Placeholder
    
    def _need_utility(self, state):
        return False  # Placeholder
    
    def _get_utility_ability(self):
        return "Utility"  # Placeholder
    
    def _get_player_target(self, player_name, state):
        return None  # Placeholder