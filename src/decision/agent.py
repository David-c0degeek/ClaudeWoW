"""
Decision Agent Module

This module handles high-level decision making and planning for the AI player.
"""

import logging
import time
import os
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np

from src.perception.screen_reader import GameState, ScreenReader
from src.action.controller import Controller
from src.decision.behavior_tree import BehaviorTree, BehaviorStatus
from src.decision.planner import Planner
from src.decision.combat_manager import CombatManager
from src.decision.navigation_manager import NavigationManager
from src.decision.navigation_integration import NavigationIntegrator
from src.decision.quest_manager import QuestManager
from src.knowledge.game_knowledge import GameKnowledge
from src.social.social_manager import SocialManager

# Import learning components
from src.learning.reinforcement_learning import ReinforcementLearningManager
from src.learning.knowledge_expansion import KnowledgeExpansionManager
from src.learning.performance_metrics import PerformanceMetricsManager
from src.learning.transfer_learning import TransferLearningManager
from src.learning.hierarchical_planning import HierarchicalPlanningManager

class Agent:
    """
    Main decision-making agent that coordinates all AI behavior
    """
    
    def __init__(self, config: Dict, screen_reader: ScreenReader, controller: Controller):
        """
        Initialize the Agent
        
        Args:
            config: Configuration dictionary
            screen_reader: ScreenReader instance for perceiving the game
            controller: Controller instance for executing actions
        """
        self.logger = logging.getLogger("wow_ai.decision.agent")
        self.config = config
        self.screen_reader = screen_reader
        self.controller = controller
        
        # Initialize game knowledge base
        self.knowledge = GameKnowledge(config)
        
        # Initialize specialized managers
        self.combat_manager = CombatManager(config, self.knowledge)
        self.navigation_manager = NavigationManager(config, self.knowledge)
        self.quest_manager = QuestManager(config, self.knowledge)
        
        # Initialize planner for high-level decision making
        self.planner = Planner(config, self.knowledge)
        
        # Initialize behavior tree
        self.behavior_tree = self._create_behavior_tree()
        
        # State variables
        self.current_state = None
        self.previous_state = None
        self.last_decision_time = 0
        self.decision_interval = config.get("decision_interval", 0.1)  # seconds

        self.social_manager = SocialManager(config, self.knowledge)
        
        # Initialize learning components if enabled
        self.learning_enabled = config.get("learning", {}).get("enabled", True)
        if self.learning_enabled:
            self._init_learning_components(config)
            self.last_save_time = time.time()
            self.save_interval = config.get("learning", {}).get("save_interval", 300)  # 5 minutes

        # Goals and plans
        self.current_goal = None
        self.current_plan = []
        self.current_plan_step = 0
        
        self.logger.info("Agent initialized")
    
    def _init_learning_components(self, config: Dict) -> None:
        """
        Initialize learning components
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Initializing learning components")
        
        # Initialize reinforcement learning
        self.rl_manager = ReinforcementLearningManager(config)
        
        # Initialize knowledge expansion
        self.knowledge_manager = KnowledgeExpansionManager(config)
        
        # Initialize performance metrics
        self.metrics_manager = PerformanceMetricsManager(config)
        
        # Initialize transfer learning
        self.transfer_manager = TransferLearningManager(config)
        
        # Create data directories if they don't exist
        os.makedirs(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        ), exist_ok=True)
        
        os.makedirs(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "game_knowledge"
        ), exist_ok=True)
        
        # Track learning-specific state
        self.last_action = None
        self.last_state_dict = None
        
        self.logger.info("Learning components initialized")
    
    def decide(self, game_state: GameState) -> List[Dict]:
        """
        Make decisions based on the current game state
        
        Args:
            game_state: Current game state from perception system
        
        Returns:
            List[Dict]: List of actions to execute
        """
        # Update state
        self.previous_state = self.current_state
        self.current_state = game_state
        
        # Check if we should make a new decision
        current_time = time.time()
        if current_time - self.last_decision_time < self.decision_interval:
            # Return the last decided actions if within interval
            return self.last_actions if hasattr(self, 'last_actions') else []
        
        self.last_decision_time = current_time
        
        try:
            # Update knowledge base with new state information
            self.knowledge.update(game_state)

            # Process observations for knowledge expansion if learning is enabled
            if self.learning_enabled:
                self._process_observations(game_state)
                
                # Record experience from previous action and state
                if hasattr(self, 'last_action') and self.last_action and self.previous_state:
                    self._record_experience(game_state)

            # Update social manager
            self.social_manager.update(game_state)
            
            # Run the behavior tree to determine high-level behavior
            behavior_status = self.behavior_tree.tick(game_state, self)
            
            if behavior_status == BehaviorStatus.FAILURE:
                self.logger.warning("Behavior tree execution failed")
                
                # Try RL-based fallback if enabled
                if self.learning_enabled and self.config.get("learning", {}).get("use_rl_for_decisions", False):
                    actions = self._make_rl_decision(game_state)
                    if actions:
                        # Store current state and action for learning
                        if self.learning_enabled:
                            self.last_action = self._action_to_string(actions)
                            self.last_state_dict = self._state_to_dict(game_state)
                        
                        self.last_actions = actions
                        return actions
                
                # Otherwise use fallback actions
                self.last_actions = self._generate_fallback_actions()
                return self.last_actions
            
            # If we have a current plan, execute the next step
            if self.current_plan and self.current_plan_step < len(self.current_plan):
                actions = self._execute_plan_step()
            else:
                # Generate a new plan if needed
                if not self.current_goal or not self.current_plan:
                    self._set_new_goal(game_state)
                    self._generate_plan()
                
                actions = self._execute_plan_step() if self.current_plan else []
            
            # Get social actions
            social_actions = self.social_manager.generate_social_actions(game_state)

            # Combine with other actions (with appropriate prioritization)
            if social_actions:
                # Check for any high-priority social actions
                high_priority_social = [a for a in social_actions if a.get("priority", 0) > 0.7]

                if high_priority_social:
                    # Prioritize high-priority social actions
                    actions = high_priority_social + actions
                else:
                    # Add social actions after regular actions
                    actions.extend(social_actions)
            
            # Store the decided actions and current state for learning
            if self.learning_enabled:
                self.last_action = self._action_to_string(actions)
                self.last_state_dict = self._state_to_dict(game_state)
                
                # Track metrics
                self._track_decision_metrics(game_state, actions)
                
                # Check if we should save learning data
                if current_time - self.last_save_time > self.save_interval:
                    self._save_learning_data()
                    self.last_save_time = current_time
            
            self.last_actions = actions
            return actions
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {e}")
            self.logger.exception(e)
            return []
    
    def _process_observations(self, game_state: GameState) -> None:
        """
        Process observations for knowledge expansion
        
        Args:
            game_state: Current game state
        """
        # Convert game state to observation format
        observation = self._state_to_observation(game_state)
        
        # Add observation to knowledge expansion system
        self.knowledge_manager.add_observation(observation)
        
        # Process a batch of observations
        self.knowledge_manager.process_observations(batch_size=5)
    
    def _state_to_observation(self, game_state: GameState) -> Dict[str, Any]:
        """
        Convert game state to observation format for knowledge expansion
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary observation
        """
        observation = {
            "timestamp": time.time(),
            "entities": [],
            "player_location": {},
            "combat": {}
        }
        
        # Add entities
        if hasattr(game_state, "nearby_entities"):
            observation["entities"] = game_state.nearby_entities
        
        # Add player location
        if hasattr(game_state, "current_zone"):
            observation["player_location"] = {
                "zone": game_state.current_zone,
                "coordinates": getattr(game_state, "coordinates", (0, 0))
            }
        
        # Add combat data if in combat
        if hasattr(game_state, "is_in_combat") and game_state.is_in_combat:
            observation["combat"] = {
                "target": getattr(game_state, "target", "unknown"),
                "health_percent": getattr(game_state, "health_percent", 100),
                "spells_used": getattr(game_state, "recent_spells", [])
            }
        
        # Add quest data
        if hasattr(game_state, "active_quests"):
            observation["quest_log"] = game_state.active_quests
        
        # Add inventory data if available
        if hasattr(game_state, "inventory"):
            observation["inventory"] = game_state.inventory
        
        return observation
    
    def _state_to_dict(self, game_state: GameState) -> Dict[str, Any]:
        """
        Convert game state to dictionary for reinforcement learning
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary representation of state
        """
        # Create a simplified state representation for RL
        state_dict = {
            "is_in_combat": getattr(game_state, "is_in_combat", False),
            "health_percent": getattr(game_state, "health_percent", 100),
            "mana_percent": getattr(game_state, "mana_percent", 100),
            "nearby_enemy_count": len([e for e in getattr(game_state, "nearby_entities", []) 
                                      if e.get("hostile", False)]),
            "is_in_group": getattr(game_state, "is_in_group", False),
            "has_target": getattr(game_state, "target", None) is not None,
            "is_resting": getattr(game_state, "is_resting", False),
            "is_moving": getattr(game_state, "is_moving", False)
        }
        
        return state_dict
    
    def _action_to_string(self, actions: List[Dict]) -> str:
        """
        Convert action list to string representation for RL
        
        Args:
            actions: List of actions
            
        Returns:
            String representation of actions
        """
        if not actions:
            return "idle"
        
        primary_action = actions[0]
        action_type = primary_action.get("type", "unknown")
        
        if action_type == "move":
            return "move"
        elif action_type == "cast":
            return f"cast_{primary_action.get('spell', 'unknown')}"
        elif action_type == "interact":
            return "interact"
        elif action_type == "use_item":
            return "use_item"
        elif action_type == "chat":
            return "chat"
        elif action_type == "jump":
            return "jump"
        elif action_type == "wait":
            return "wait"
        else:
            return action_type
    
    def _record_experience(self, game_state: GameState) -> None:
        """
        Record learning experience from previous action and current results
        
        Args:
            game_state: Current game state
        """
        # Calculate reward based on state transitions
        reward = self._calculate_reward(self.previous_state, game_state)
        
        # Add to reinforcement learning manager
        self.rl_manager.record_experience(
            self.last_state_dict, 
            self.last_action, 
            reward, 
            self._state_to_dict(game_state)
        )
    
    def _calculate_reward(self, previous_state: GameState, current_state: GameState) -> float:
        """
        Calculate reward based on state transition
        
        Args:
            previous_state: Previous game state
            current_state: Current game state
            
        Returns:
            Reward value
        """
        if not previous_state:
            return 0.0
        
        reward = 0.0
        
        # Reward for staying alive
        reward += 0.01
        
        # Reward for health changes
        prev_health = getattr(previous_state, "health_percent", 100)
        curr_health = getattr(current_state, "health_percent", 100)
        health_change = curr_health - prev_health
        
        # Penalize health loss, reward health gain
        reward += health_change * 0.05
        
        # Reward for quest progress
        if hasattr(current_state, "quest_progress") and hasattr(previous_state, "quest_progress"):
            if current_state.quest_progress > previous_state.quest_progress:
                reward += 1.0
        
        # Reward for killing enemies
        if (getattr(previous_state, "is_in_combat", False) and 
            not getattr(current_state, "is_in_combat", True) and
            getattr(previous_state, "target", None) is not None and
            getattr(current_state, "target", None) is None):
            # Combat ended and target is gone - likely killed an enemy
            reward += 2.0
        
        # Penalize death
        if (getattr(previous_state, "is_alive", True) and
            not getattr(current_state, "is_alive", True)):
            reward -= 10.0
        
        # Reward for XP gain
        prev_xp = getattr(previous_state, "xp", 0)
        curr_xp = getattr(current_state, "xp", 0)
        if curr_xp > prev_xp:
            reward += 0.5
        
        # Reward for level up
        prev_level = getattr(previous_state, "level", 1)
        curr_level = getattr(current_state, "level", 1)
        if curr_level > prev_level:
            reward += 5.0
        
        return reward
    
    def _make_rl_decision(self, game_state: GameState) -> List[Dict]:
        """
        Make a decision using reinforcement learning
        
        Args:
            game_state: Current game state
            
        Returns:
            List of actions to take
        """
        # Convert state to RL-friendly format
        state_dict = self._state_to_dict(game_state)
        
        # Get all possible action types
        available_actions = self._get_available_actions(game_state)
        
        if not available_actions:
            return []
        
        # Let RL choose an action
        chosen_action = self.rl_manager.choose_action(state_dict, available_actions)
        
        # Convert the chosen action back to concrete actions
        return self._action_string_to_actions(chosen_action, game_state)
    
    def _get_available_actions(self, game_state: GameState) -> List[str]:
        """
        Get available actions for the current state
        
        Args:
            game_state: Current game state
            
        Returns:
            List of available action strings
        """
        # Basic actions always available
        actions = ["idle", "wait", "look"]
        
        # Movement is usually available
        actions.append("move")
        
        # Add jump if not in combat
        if not getattr(game_state, "is_in_combat", False):
            actions.append("jump")
        
        # Combat actions if in combat
        if getattr(game_state, "is_in_combat", False):
            # Add generic combat actions
            actions.append("cast_attack")
            
            # Add class-specific spells if available
            player_class = getattr(game_state, "player_class", "").lower()
            if player_class == "warrior":
                actions.extend(["cast_charge", "cast_rend", "cast_thunderclap"])
            elif player_class == "mage":
                actions.extend(["cast_fireball", "cast_frostbolt", "cast_blink"])
            # Add other classes as needed
        
        # Add interaction if there are nearby interactable entities
        if any(e.get("interactable", False) for e in getattr(game_state, "nearby_entities", [])):
            actions.append("interact")
        
        # Add item use if there are usable items
        if getattr(game_state, "usable_items", []):
            actions.append("use_item")
        
        return actions
    
    def _action_string_to_actions(self, action_string: str, game_state: GameState) -> List[Dict]:
        """
        Convert action string to concrete actions
        
        Args:
            action_string: Action string from RL
            game_state: Current game state
            
        Returns:
            List of concrete actions
        """
        actions = []
        
        if action_string == "idle":
            actions.append({"type": "wait", "duration": 0.5})
        
        elif action_string == "wait":
            actions.append({"type": "wait", "duration": 1.0})
        
        elif action_string == "move":
            # Choose a reasonable movement direction
            # For RL, we might want to move toward objectives or away from danger
            if getattr(game_state, "is_in_combat", False):
                # In combat, maybe back up from enemy
                actions.append({
                    "type": "move",
                    "x": random.uniform(-1, 0),
                    "y": random.uniform(-1, 0),
                    "duration": 0.5
                })
            else:
                # Exploration movement
                actions.append({
                    "type": "move",
                    "x": random.uniform(-1, 1),
                    "y": random.uniform(-1, 1),
                    "duration": 0.5
                })
        
        elif action_string == "jump":
            actions.append({"type": "jump"})
        
        elif action_string.startswith("cast_"):
            spell = action_string.replace("cast_", "")
            target = getattr(game_state, "target", None)
            actions.append({
                "type": "cast",
                "spell": spell,
                "target": target,
                "duration": 0.1
            })
        
        elif action_string == "interact":
            # Find nearest interactable entity
            interactables = [e for e in getattr(game_state, "nearby_entities", []) 
                            if e.get("interactable", False)]
            if interactables:
                nearest = interactables[0]  # Assume first is nearest for simplicity
                actions.append({
                    "type": "interact",
                    "target": nearest.get("id"),
                    "duration": 0.1
                })
        
        elif action_string == "use_item":
            # Use first available item
            usable_items = getattr(game_state, "usable_items", [])
            if usable_items:
                actions.append({
                    "type": "use_item",
                    "item": usable_items[0],
                    "duration": 0.1
                })
        
        return actions
    
    def _track_decision_metrics(self, game_state: GameState, actions: List[Dict]) -> None:
        """
        Track metrics related to decision making
        
        Args:
            game_state: Current game state
            actions: Decided actions
        """
        # Track decision time
        decision_time = time.time() - self.last_decision_time
        self.metrics_manager.record_metric("decision_time", decision_time * 1000)  # Convert to ms
        
        # Track action distribution
        if actions:
            action_type = actions[0].get("type", "unknown")
            metric_name = f"action_distribution_{action_type}"
            self.metrics_manager.record_metric(metric_name, 1.0)
        
        # Track combat metrics if in combat
        if getattr(game_state, "is_in_combat", False):
            # DPS if available
            if hasattr(game_state, "current_dps"):
                self.metrics_manager.record_metric("combat_dps", game_state.current_dps)
            
            # Resource efficiency
            if hasattr(game_state, "resource_efficiency"):
                self.metrics_manager.record_metric(
                    "combat_resource_efficiency", 
                    game_state.resource_efficiency
                )
    
    def _save_learning_data(self) -> None:
        """Save all learning data to disk"""
        try:
            # Save reinforcement learning data
            self.rl_manager.save_models()
            
            # Save knowledge expansion data
            self.knowledge_manager.save_knowledge_base()
            
            # Save performance metrics
            self.metrics_manager.save_metrics()
            
            # Save transfer learning skills
            self.transfer_manager.save_skills()
            
            self.logger.info("Saved all learning data")
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
    
    def learn_from_batch(self, batch_size: int = 64) -> None:
        """
        Trigger learning from collected experiences
        
        Args:
            batch_size: Number of experiences to learn from
        """
        if not self.learning_enabled:
            return
            
        self.rl_manager.learn_from_batch(batch_size)
        self.logger.info(f"Learned from batch of {batch_size} experiences")
    
    def register_transferable_skill(self, skill_name: str, description: str, 
                                    context: str, parameters: Dict[str, Any] = None) -> None:
        """
        Register a new transferable skill
        
        Args:
            skill_name: Name of the skill
            description: Skill description
            context: Source context
            parameters: Skill parameters
        """
        if not self.learning_enabled:
            return
            
        skill = TransferableSkill(
            name=skill_name,
            description=description,
            source_context=context,
            parameters=parameters or {}
        )
        
        self.transfer_manager.register_skill(skill)
        self.logger.info(f"Registered transferable skill: {skill_name}")
    
    def generate_performance_report(self, output_path: str = None) -> str:
        """
        Generate a performance report
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report text
        """
        if not self.learning_enabled:
            return "Learning is disabled"
            
        return self.metrics_manager.generate_performance_report(output_path)
    
    def _create_behavior_tree(self) -> BehaviorTree:
        """
        Create the behavior tree for decision making
        
        Returns:
            BehaviorTree: Initialized behavior tree
        """
        # In a real implementation, we would build a complex behavior tree here
        # This is a simplified placeholder
        behavior_tree = BehaviorTree()
        
        # Add nodes to the tree
        behavior_tree.add_selector("root")
        
        # Add combat sequence
        behavior_tree.add_sequence("combat", parent="root")
        behavior_tree.add_condition("is_in_combat", parent="combat", 
                                  condition_func=lambda state, agent: state.is_in_combat)
        behavior_tree.add_action("handle_combat", parent="combat", 
                               action_func=self._handle_combat)
        
        # Add quest sequence
        behavior_tree.add_sequence("quest", parent="root")
        behavior_tree.add_condition("has_active_quest", parent="quest",
                                  condition_func=lambda state, agent: len(state.active_quests) > 0)
        behavior_tree.add_action("handle_quest", parent="quest",
                               action_func=self._handle_quest)
        
        # Add exploration sequence
        behavior_tree.add_sequence("explore", parent="root")
        behavior_tree.add_action("handle_exploration", parent="explore",
                               action_func=self._handle_exploration)
        
        return behavior_tree
    
    def _handle_combat(self, state: GameState, agent: 'Agent') -> BehaviorStatus:
        """
        Handle combat situations
        
        Args:
            state: Current game state
            agent: The agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        self.logger.info("Handling combat situation")
        
        # Set combat as the current goal
        self.current_goal = {
            "type": "combat",
            "target": state.target,
            "priority": 1.0
        }
        
        # Generate a combat plan
        combat_plan = self.combat_manager.generate_combat_plan(state)
        
        if combat_plan:
            self.current_plan = combat_plan
            self.current_plan_step = 0
            return BehaviorStatus.SUCCESS
        else:
            return BehaviorStatus.FAILURE
    
    def _handle_quest(self, state: GameState, agent: 'Agent') -> BehaviorStatus:
        """
        Handle quest-related activities
        
        Args:
            state: Current game state
            agent: The agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        self.logger.info("Handling quest activities")
        
        # Set quest progression as the current goal
        if state.current_quest:
            self.current_goal = {
                "type": "quest",
                "quest": state.current_quest,
                "priority": 0.8
            }
            
            # Generate a quest plan
            quest_plan = self.quest_manager.generate_quest_plan(state)
            
            if quest_plan:
                self.current_plan = quest_plan
                self.current_plan_step = 0
                return BehaviorStatus.SUCCESS
        
        return BehaviorStatus.FAILURE
    
    def _handle_exploration(self, state: GameState, agent: 'Agent') -> BehaviorStatus:
        """
        Handle exploration and idle activities
        
        Args:
            state: Current game state
            agent: The agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        self.logger.info("Handling exploration")
        
        # Set exploration as the current goal
        self.current_goal = {
            "type": "exploration",
            "priority": 0.5
        }
        
        # Generate an exploration plan
        exploration_plan = self.navigation_manager.generate_exploration_plan(state)
        
        if exploration_plan:
            self.current_plan = exploration_plan
            self.current_plan_step = 0
            return BehaviorStatus.SUCCESS
        else:
            return BehaviorStatus.FAILURE
    
    def _set_new_goal(self, state: GameState) -> None:
        """
        Set a new goal based on the current state
        
        Args:
            state: Current game state
        """
        # Priority queue of potential goals
        potential_goals = []
        
        # Combat goal
        if state.is_in_combat:
            potential_goals.append({
                "type": "combat",
                "target": state.target,
                "priority": 1.0
            })
        
        # Quest goal
        if state.active_quests:
            potential_goals.append({
                "type": "quest",
                "quest": state.current_quest,
                "priority": 0.8
            })
        
        # Loot goal (if there are nearby lootable corpses)
        nearby_lootable = any(e.get("type") == "lootable" for e in state.nearby_entities)
        if nearby_lootable:
            potential_goals.append({
                "type": "loot",
                "priority": 0.7
            })
        
        # Vendor goal (if inventory is getting full)
        inventory_full = False  # This would be determined from state in a real implementation
        if inventory_full:
            potential_goals.append({
                "type": "vendor",
                "priority": 0.6
            })
        
        # Exploration goal (fallback)
        potential_goals.append({
            "type": "exploration",
            "priority": 0.5
        })
        
        # Select highest priority goal
        if potential_goals:
            self.current_goal = max(potential_goals, key=lambda g: g["priority"])
        else:
            self.current_goal = {"type": "idle", "priority": 0.1}
    
    def _generate_plan(self) -> None:
        """
        Generate a plan to achieve the current goal
        """
        if not self.current_goal:
            self.logger.warning("Attempted to generate plan without a goal")
            return
        
        plan = []
        
        # Generate plan based on goal type
        goal_type = self.current_goal.get("type")
        
        if goal_type == "combat":
            plan = self.combat_manager.generate_combat_plan(self.current_state)
        elif goal_type == "quest":
            plan = self.quest_manager.generate_quest_plan(self.current_state)
        elif goal_type == "loot":
            plan = self._generate_loot_plan()
        elif goal_type == "vendor":
            plan = self._generate_vendor_plan()
        elif goal_type == "exploration":
            plan = self.navigation_manager.generate_exploration_plan(self.current_state)
        else:
            plan = self._generate_idle_plan()
        
        self.current_plan = plan
        self.current_plan_step = 0
        
        self.logger.info(f"Generated plan with {len(plan)} steps for goal type: {goal_type}")
    
    def _execute_plan_step(self) -> List[Dict]:
        """
        Execute the current step in the plan
        
        Returns:
            List[Dict]: Actions to execute
        """
        if not self.current_plan or self.current_plan_step >= len(self.current_plan):
            return []
        
        # Get the current step
        step = self.current_plan[self.current_plan_step]
        
        # Execute the step
        actions = []
        
        # Convert the plan step to concrete actions
        if step.get("type") == "move":
            # Convert movement to specific actions
            target_pos = step.get("position")
            if target_pos:
                actions.append({
                    "type": "move",
                    "x": target_pos[0],
                    "y": target_pos[1],
                    "duration": 0.5
                })
        
        elif step.get("type") == "interact":
            # Convert interaction to specific actions
            target = step.get("target")
            if target:
                actions.append({
                    "type": "interact",
                    "target": target,
                    "duration": 0.1
                })
        
        elif step.get("type") == "cast":
            # Convert spell cast to specific actions
            spell = step.get("spell")
            target = step.get("target")
            if spell:
                actions.append({
                    "type": "cast",
                    "spell": spell,
                    "target": target,
                    "duration": 0.1
                })
        
        elif step.get("type") == "use_item":
            # Convert item use to specific actions
            item = step.get("item")
            if item:
                actions.append({
                    "type": "use_item",
                    "item": item,
                    "duration": 0.1
                })
        
        # Advance to the next step
        self.current_plan_step += 1
        
        # Check if the plan is complete
        if self.current_plan_step >= len(self.current_plan):
            self.logger.info("Plan execution completed")
            # Clear the current plan to generate a new one next time
            self.current_plan = []
            self.current_plan_step = 0
        
        return actions
    
    def _generate_loot_plan(self) -> List[Dict]:
        """
        Generate a plan for looting nearby corpses
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        
        # Find lootable entities
        lootable_entities = [e for e in self.current_state.nearby_entities 
                             if e.get("type") == "lootable"]
        
        for entity in lootable_entities:
            # Add movement to the entity
            plan.append({
                "type": "move",
                "position": entity.get("position"),
                "description": f"Move to lootable {entity.get('id')}"
            })
            
            # Add interaction with the entity
            plan.append({
                "type": "interact",
                "target": entity.get("id"),
                "description": f"Loot {entity.get('id')}"
            })
        
        return plan
    
    def _generate_vendor_plan(self) -> List[Dict]:
        """
        Generate a plan for visiting a vendor
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        
        # Find nearest vendor
        vendor_entities = [e for e in self.current_state.nearby_entities 
                           if e.get("subtype") == "vendor"]
        
        if vendor_entities:
            vendor = vendor_entities[0]  # Choose the first vendor
            
            # Add movement to the vendor
            plan.append({
                "type": "move",
                "position": vendor.get("position"),
                "description": f"Move to vendor {vendor.get('id')}"
            })
            
            # Add interaction with the vendor
            plan.append({
                "type": "interact",
                "target": vendor.get("id"),
                "description": f"Talk to vendor {vendor.get('id')}"
            })
            
            # Add steps for selling items
            plan.append({
                "type": "sell_items",
                "items": ["junk", "gray_items", "excess_items"],
                "description": "Sell junk and unwanted items"
            })
            
            # Add step for repairing gear
            plan.append({
                "type": "repair",
                "description": "Repair all gear"
            })
        
        return plan
    
    def _generate_idle_plan(self) -> List[Dict]:
        """
        Generate a plan for idle behavior
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        
        # Add simple idle behaviors
        idle_actions = [
            {"type": "emote", "emote": "dance", "description": "Dance"},
            {"type": "emote", "emote": "sit", "description": "Sit down"},
            {"type": "emote", "emote": "wave", "description": "Wave"},
            {"type": "jump", "description": "Jump"},
            {"type": "wait", "duration": 2.0, "description": "Wait"}
        ]
        
        # Choose a random idle action
        if idle_actions:
            plan.append(random.choice(idle_actions))
        
        return plan
    
    def _generate_fallback_actions(self) -> List[Dict]:
        """
        Generate fallback actions when behavior tree fails
        
        Returns:
            List[Dict]: Fallback actions
        """
        # Simple random actions as fallback
        possible_actions = [
            {"type": "move", "x": random.uniform(-1, 1), "y": random.uniform(-1, 1), "duration": 0.5},
            {"type": "jump"},
            {"type": "turn", "angle": random.uniform(-30, 30)},
            {"type": "wait", "duration": 0.5}
        ]
        
        # Choose 1-2 random actions
        num_actions = random.randint(1, 2)
        return random.sample(possible_actions, num_actions)