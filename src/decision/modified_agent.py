"""
Modified Decision Agent Module

This module extends the basic agent with advanced navigation capabilities.
"""

import logging
import time
import os
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np

from src.perception.screen_reader import GameState, ScreenReader
from src.action.controller import Controller
from src.action.movement_controller import MovementController
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

class ModifiedAgent:
    """
    Main decision-making agent with advanced navigation capabilities
    """
    
    def __init__(self, config: Dict, screen_reader: ScreenReader, controller: Controller):
        """
        Initialize the Agent
        
        Args:
            config: Configuration dictionary
            screen_reader: ScreenReader instance for perceiving the game
            controller: Controller instance for executing actions
        """
        self.logger = logging.getLogger("wow_ai.decision.modified_agent")
        self.config = config
        self.screen_reader = screen_reader
        self.controller = controller
        
        # Initialize game knowledge base
        self.knowledge = GameKnowledge(config)
        
        # Initialize specialized controllers
        self.movement_controller = MovementController(config, controller)
        
        # Initialize specialized managers
        self.combat_manager = CombatManager(config, self.knowledge)
        self.navigation_manager = NavigationIntegrator(config, self.knowledge)
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
        
        self.logger.info("ModifiedAgent initialized with advanced navigation")
    
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
        
        # Initialize hierarchical planning
        self.hierarchical_planner = HierarchicalPlanningManager(config)
        
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
                self.last_actions = self._generate_fallback_actions(game_state)
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
            
            # Execute movement-based actions using movement controller
            if actions and actions[0].get("type") in ["move", "jump", "turn", "strafe"]:
                # Extract the first movement action
                movement_action = actions[0]
                
                # Execute movement with movement controller
                completion = self.movement_controller.execute_movement(movement_action, game_state)
                
                # If movement is complete, remove it from actions
                if completion:
                    actions = actions[1:]
            
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
            "combat": {},
            "terrain": {}
        }
        
        # Add entities
        if hasattr(game_state, "nearby_entities"):
            observation["entities"] = game_state.nearby_entities
        
        # Add player location
        if hasattr(game_state, "current_zone"):
            observation["player_location"] = {
                "zone": game_state.current_zone,
                "coordinates": getattr(game_state, "player_position", (0, 0, 0))
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
        
        # Add terrain data if available
        if hasattr(game_state, "terrain_data"):
            observation["terrain"] = game_state.terrain_data
        
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
            "is_moving": getattr(game_state, "is_moving", False),
            "current_zone": getattr(game_state, "current_zone", "unknown"),
            "in_dungeon": getattr(game_state, "in_dungeon", False),
            "is_mounted": getattr(game_state, "is_mounted", False),
            "elevation": getattr(game_state, "player_position", (0, 0, 0))[2] if 
                        len(getattr(game_state, "player_position", (0, 0))) > 2 else 0
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
        elif action_type == "use_flight_path":
            return "use_flight_path"
        elif action_type == "mount":
            return "mount"
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
        
        # Rewards for navigation success
        # Getting to a destination faster than expected
        if hasattr(self.navigation_manager, "current_destination") and self.navigation_manager.current_destination:
            if self.navigation_manager._has_reached_destination(current_state):
                reward += 1.0
        
        # Reward for discovering new areas
        if (hasattr(previous_state, "discovered_areas") and hasattr(current_state, "discovered_areas") and
            len(current_state.discovered_areas) > len(previous_state.discovered_areas)):
            reward += 0.5
        
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
        
        # Add mount/dismount if appropriate
        if getattr(game_state, "can_mount", False) and not getattr(game_state, "is_mounted", False):
            actions.append("mount")
        elif getattr(game_state, "is_mounted", False):
            actions.append("dismount")
            
        # Add flight path if near a flight master
        if getattr(game_state, "nearest_npc_type", "") == "flight_master":
            actions.append("use_flight_path")
        
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
            # Use navigation system to determine a good place to move
            if hasattr(game_state, "player_position"):
                current_pos = game_state.player_position
                
                # Determine a target position (this is simplified)
                # In a real system, this would choose a meaningful destination
                if hasattr(game_state, "points_of_interest") and game_state.points_of_interest:
                    # Move toward a point of interest
                    target_poi = game_state.points_of_interest[0]
                    actions = self.navigation_manager.navigate_to(
                        game_state, 
                        target_poi.get("position"), 
                        target_poi.get("zone")
                    )
                else:
                    # Generate exploration movement
                    target_x = current_pos[0] + random.uniform(-50, 50)
                    target_y = current_pos[1] + random.uniform(-50, 50)
                    target_z = current_pos[2] if len(current_pos) > 2 else 0
                    
                    actions = self.navigation_manager.navigate_to(
                        game_state, 
                        (target_x, target_y, target_z),
                        getattr(game_state, "current_zone", None)
                    )
        
        elif action_string == "jump":
            actions.append({"type": "jump"})
        
        elif action_string == "mount":
            actions.append({"type": "mount"})
        
        elif action_string == "dismount":
            actions.append({"type": "dismount"})
            
        elif action_string == "use_flight_path":
            # Get available flight paths and choose one
            flight_paths = self.navigation_manager.get_flight_paths(game_state)
            
            if flight_paths:
                # Choose a random destination for exploration
                destination = random.choice(flight_paths)
                dest_pos = destination.get("position", (0, 0, 0))
                dest_zone = destination.get("zone", "")
                
                actions = self.navigation_manager.navigate_to(
                    game_state,
                    dest_pos,
                    dest_zone,
                    mode="flight"
                )
        
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
        
        # Track navigation metrics
        if hasattr(self.navigation_manager, "current_path") and self.navigation_manager.current_path:
            self.metrics_manager.record_metric("navigation_path_length", len(self.navigation_manager.current_path))
        
        # Track if we're using advanced navigation
        self.metrics_manager.record_metric(
            f"navigation_mode_{self.navigation_manager.current_navigation_mode}", 1.0
        )
        
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
            
        skill = {
            "name": skill_name,
            "description": description,
            "source_context": context,
            "parameters": parameters or {}
        }
        
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
        # Create a more advanced behavior tree
        behavior_tree = BehaviorTree()
        
        # Add nodes to the tree
        behavior_tree.add_selector("root")
        
        # Add emergency sequence (highest priority)
        behavior_tree.add_sequence("emergency", parent="root")
        behavior_tree.add_condition("is_low_health", parent="emergency", 
                                  condition_func=lambda state, agent: getattr(state, "health_percent", 100) < 25)
        behavior_tree.add_action("handle_emergency", parent="emergency", 
                               action_func=self._handle_emergency)
        
        # Add combat sequence
        behavior_tree.add_sequence("combat", parent="root")
        behavior_tree.add_condition("is_in_combat", parent="combat", 
                                  condition_func=lambda state, agent: getattr(state, "is_in_combat", False))
        behavior_tree.add_action("handle_combat", parent="combat", 
                               action_func=self._handle_combat)
        
        # Add dungeon sequence
        behavior_tree.add_sequence("dungeon", parent="root")
        behavior_tree.add_condition("is_in_dungeon", parent="dungeon",
                                  condition_func=lambda state, agent: getattr(state, "in_dungeon", False))
        behavior_tree.add_action("handle_dungeon", parent="dungeon",
                               action_func=self._handle_dungeon)
        
        # Add quest sequence
        behavior_tree.add_sequence("quest", parent="root")
        behavior_tree.add_condition("has_active_quest", parent="quest",
                                  condition_func=lambda state, agent: len(getattr(state, "active_quests", [])) > 0)
        behavior_tree.add_action("handle_quest", parent="quest",
                               action_func=self._handle_quest)
        
        # Add travel sequence
        behavior_tree.add_sequence("travel", parent="root")
        behavior_tree.add_condition("needs_travel", parent="travel",
                                  condition_func=lambda state, agent: hasattr(agent, "travel_destination") and
                                 agent.travel_destination is not None)
        behavior_tree.add_action("handle_travel", parent="travel",
                               action_func=self._handle_travel)
        
        # Add exploration sequence
        behavior_tree.add_sequence("explore", parent="root")
        behavior_tree.add_action("handle_exploration", parent="explore",
                               action_func=self._handle_exploration)
        
        return behavior_tree
    
    def _handle_emergency(self, state: GameState, agent: 'ModifiedAgent') -> BehaviorStatus:
        """
        Handle emergency situations (low health, etc.)
        
        Args:
            state: Current game state
            agent: The agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        self.logger.info("Handling emergency situation")
        
        # Set emergency handling as the current goal
        self.current_goal = {
            "type": "emergency",
            "priority": 1.0
        }
        
        # Create an emergency plan
        emergency_plan = []
        
        # Check if we have healing items
        if hasattr(state, "healing_items") and state.healing_items:
            # Use a healing item
            emergency_plan.append({
                "type": "use_item",
                "item": state.healing_items[0],
                "description": "Use healing item"
            })
        
        # Check if we should run away
        if getattr(state, "is_in_combat", False) and getattr(state, "health_percent", 100) < 15:
            # Try to escape combat
            if hasattr(state, "player_position") and hasattr(state, "nearby_entities"):
                # Find the farthest point from enemies
                enemies = [e for e in state.nearby_entities if e.get("hostile", False)]
                
                if enemies:
                    # Calculate average enemy position
                    avg_x = sum(e.get("position", [0, 0])[0] for e in enemies) / len(enemies)
                    avg_y = sum(e.get("position", [0, 0])[1] for e in enemies) / len(enemies)
                    
                    # Calculate a point in the opposite direction
                    direction_x = state.player_position[0] - avg_x
                    direction_y = state.player_position[1] - avg_y
                    
                    # Normalize and scale the direction
                    magnitude = (direction_x**2 + direction_y**2)**0.5
                    if magnitude > 0:
                        direction_x = direction_x / magnitude * 50  # Run 50 units away
                        direction_y = direction_y / magnitude * 50
                    
                    # Create a flee position
                    flee_x = state.player_position[0] + direction_x
                    flee_y = state.player_position[1] + direction_y
                    flee_z = state.player_position[2] if len(state.player_position) > 2 else 0
                    
                    # Add fleeing navigation action
                    emergency_actions = self.navigation_manager.navigate_to(
                        state, (flee_x, flee_y, flee_z), getattr(state, "current_zone", None)
                    )
                    
                    emergency_plan.extend(emergency_actions)
        
        if emergency_plan:
            self.current_plan = emergency_plan
            self.current_plan_step = 0
            return BehaviorStatus.SUCCESS
        else:
            return BehaviorStatus.FAILURE
    
    def _handle_combat(self, state: GameState, agent: 'ModifiedAgent') -> BehaviorStatus:
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
            "target": getattr(state, "target", None),
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
    
    def _handle_dungeon(self, state: GameState, agent: 'ModifiedAgent') -> BehaviorStatus:
        """
        Handle dungeon navigation and objectives
        
        Args:
            state: Current game state
            agent: The agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        self.logger.info("Handling dungeon situation")
        
        # Set dungeon navigation as the current goal
        self.current_goal = {
            "type": "dungeon",
            "priority": 0.9
        }
        
        # Get the next boss to target
        next_boss = self.navigation_manager.navigation_manager.dungeon_navigator.get_next_boss(state)
        
        if next_boss:
            # Create a plan to navigate to the boss
            boss_name = next_boss.get("name", "")
            dungeon_actions = self.navigation_manager.navigation_manager.dungeon_navigator.navigate_to_boss(
                state, boss_name
            )
            
            if dungeon_actions:
                self.current_plan = dungeon_actions
                self.current_plan_step = 0
                return BehaviorStatus.SUCCESS
        
        # If no boss found, explore the dungeon
        # Find the nearest point of interest
        nearest_poi = self.navigation_manager.navigation_manager.dungeon_navigator.find_nearest_point_of_interest(state)
        
        if nearest_poi:
            # Navigate to the POI
            poi_pos = nearest_poi.get("position")
            poi_area = nearest_poi.get("area")
            
            if poi_pos:
                # Create a plan to navigate to the POI
                dungeon_actions = self.navigation_manager.navigate_to(
                    state, poi_pos, poi_area
                )
                
                if dungeon_actions:
                    self.current_plan = dungeon_actions
                    self.current_plan_step = 0
                    return BehaviorStatus.SUCCESS
        
        return BehaviorStatus.FAILURE
    
    def _handle_quest(self, state: GameState, agent: 'ModifiedAgent') -> BehaviorStatus:
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
        current_quest = getattr(state, "current_quest", None)
        if current_quest:
            self.current_goal = {
                "type": "quest",
                "quest": current_quest,
                "priority": 0.8
            }
            
            # Generate a quest plan
            quest_plan = self.quest_manager.generate_quest_plan(state)
            
            if quest_plan:
                self.current_plan = quest_plan
                self.current_plan_step = 0
                return BehaviorStatus.SUCCESS
        
        return BehaviorStatus.FAILURE
    
    def _handle_travel(self, state: GameState, agent: 'ModifiedAgent') -> BehaviorStatus:
        """
        Handle travel to a destination
        
        Args:
            state: Current game state
            agent: The agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        self.logger.info("Handling travel")
        
        if not hasattr(agent, "travel_destination") or agent.travel_destination is None:
            return BehaviorStatus.FAILURE
        
        # Extract destination information
        destination = agent.travel_destination
        destination_pos = destination.get("position")
        destination_zone = destination.get("zone")
        
        if not destination_pos:
            return BehaviorStatus.FAILURE
        
        # Set travel as the current goal
        self.current_goal = {
            "type": "travel",
            "destination": destination,
            "priority": 0.7
        }
        
        # Use the navigation integrator to generate a route
        travel_actions = self.navigation_manager.navigate_to(
            state, destination_pos, destination_zone
        )
        
        if travel_actions:
            self.current_plan = travel_actions
            self.current_plan_step = 0
            return BehaviorStatus.SUCCESS
        else:
            return BehaviorStatus.FAILURE
    
    def _handle_exploration(self, state: GameState, agent: 'ModifiedAgent') -> BehaviorStatus:
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
        
        # Use terrain analysis to identify interesting areas to explore
        terrain_analysis = self.navigation_manager.analyze_terrain(state)
        
        # Generate an exploration plan
        if hasattr(state, "player_position"):
            current_pos = state.player_position
            current_zone = getattr(state, "current_zone", None)
            
            # Try to find an interesting feature to explore
            exploration_target = None
            
            # Check for points of interest in the terrain analysis
            if terrain_analysis.get("jump_paths"):
                # Explore a jump path
                jump_path = terrain_analysis["jump_paths"][0]
                exploration_target = jump_path["end"]
            
            # If no jump paths, try to find elevation changes or other features
            if not exploration_target and terrain_analysis.get("terrain_samples"):
                # Find a walkable high point to explore
                walkable_samples = [s for s in terrain_analysis["terrain_samples"] if s.get("walkable", False)]
                if walkable_samples:
                    # Sort by elevation (highest first)
                    walkable_samples.sort(key=lambda s: s["position"][2] if len(s["position"]) > 2 else 0, reverse=True)
                    exploration_target = walkable_samples[0]["position"]
            
            # If still no target, use basic exploration
            if not exploration_target:
                # Generate a random exploration target
                target_x = current_pos[0] + random.uniform(-100, 100)
                target_y = current_pos[1] + random.uniform(-100, 100)
                target_z = current_pos[2] if len(current_pos) > 2 else 0
                
                exploration_target = (target_x, target_y, target_z)
            
            # Navigate to the exploration target
            exploration_actions = self.navigation_manager.navigate_to(
                state, exploration_target, current_zone
            )
            
            if exploration_actions:
                self.current_plan = exploration_actions
                self.current_plan_step = 0
                return BehaviorStatus.SUCCESS
            
        # Default to basic navigation's exploration plan if our advanced exploration failed
        basic_plan = self.navigation_manager.basic_navigation.generate_exploration_plan(state)
        
        if basic_plan:
            self.current_plan = basic_plan
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
        
        # Emergency goal (highest priority)
        if getattr(state, "health_percent", 100) < 25:
            potential_goals.append({
                "type": "emergency",
                "priority": 1.1
            })
        
        # Combat goal
        if getattr(state, "is_in_combat", False):
            potential_goals.append({
                "type": "combat",
                "target": getattr(state, "target", None),
                "priority": 1.0
            })
        
        # Dungeon goal
        if getattr(state, "in_dungeon", False):
            potential_goals.append({
                "type": "dungeon",
                "priority": 0.9
            })
        
        # Quest goal
        if getattr(state, "active_quests", []):
            potential_goals.append({
                "type": "quest",
                "quest": getattr(state, "current_quest", None),
                "priority": 0.8
            })
        
        # Travel goal (if we have a destination set)
        if hasattr(self, "travel_destination") and self.travel_destination:
            potential_goals.append({
                "type": "travel",
                "destination": self.travel_destination,
                "priority": 0.7
            })
        
        # Loot goal (if there are nearby lootable corpses)
        nearby_lootable = any(e.get("type") == "lootable" for e in getattr(state, "nearby_entities", []))
        if nearby_lootable:
            potential_goals.append({
                "type": "loot",
                "priority": 0.6
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
        Generate a plan to achieve the current goal using hierarchical planning
        """
        if not self.current_goal:
            self.logger.warning("Attempted to generate plan without a goal")
            return
        
        # If learning enabled, use hierarchical planning
        if self.learning_enabled:
            plan = self.hierarchical_planner.generate_plan(self.current_goal, self.current_state)
            
            if plan:
                self.current_plan = plan
                self.current_plan_step = 0
                self.logger.info(f"Generated hierarchical plan with {len(plan)} steps for goal: {self.current_goal['type']}")
                return
        
        # Fallback to regular planning if hierarchical planning fails or learning disabled
        plan = []
        
        # Generate plan based on goal type
        goal_type = self.current_goal.get("type")
        
        if goal_type == "emergency":
            # Create plan for emergency situations
            plan = self._generate_emergency_plan()
        elif goal_type == "combat":
            plan = self.combat_manager.generate_combat_plan(self.current_state)
        elif goal_type == "dungeon":
            # Get the next boss
            next_boss = self.navigation_manager.navigation_manager.dungeon_navigator.get_next_boss(self.current_state)
            if next_boss:
                plan = self.navigation_manager.navigation_manager.dungeon_navigator.navigate_to_boss(
                    self.current_state, next_boss.get("name", "")
                )
        elif goal_type == "quest":
            plan = self.quest_manager.generate_quest_plan(self.current_state)
        elif goal_type == "travel":
            # Get destination info
            destination = self.current_goal.get("destination", {})
            dest_pos = destination.get("position")
            dest_zone = destination.get("zone")
            
            if dest_pos:
                plan = self.navigation_manager.navigate_to(
                    self.current_state, dest_pos, dest_zone
                )
        elif goal_type == "loot":
            plan = self._generate_loot_plan()
        elif goal_type == "vendor":
            plan = self._generate_vendor_plan()
        elif goal_type == "exploration":
            # Use advanced exploration
            plan = self._generate_exploration_plan()
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
                    "position": target_pos,
                    "description": step.get("description", "Move to position")
                })
        
        elif step.get("type") == "interact":
            # Convert interaction to specific actions
            target = step.get("target")
            if target:
                actions.append({
                    "type": "interact",
                    "target": target,
                    "description": step.get("description", "Interact with target")
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
                    "description": step.get("description", f"Cast {spell}")
                })
        
        elif step.get("type") == "use_item":
            # Convert item use to specific actions
            item = step.get("item")
            if item:
                actions.append({
                    "type": "use_item",
                    "item": item,
                    "description": step.get("description", f"Use {item}")
                })
                
        elif step.get("type") == "use_flight_path":
            # Convert flight path use to specific actions
            source = step.get("source")
            destination = step.get("destination")
            if source and destination:
                actions.append({
                    "type": "use_flight_path",
                    "source": source,
                    "destination": destination,
                    "description": step.get("description", f"Fly from {source} to {destination}")
                })
                
        elif step.get("type") == "mount":
            # Convert mount command to specific actions
            actions.append({
                "type": "mount",
                "description": step.get("description", "Mount up")
            })
            
        elif step.get("type") == "dismount":
            # Convert dismount command to specific actions
            actions.append({
                "type": "dismount",
                "description": step.get("description", "Dismount")
            })
            
        elif step.get("type") == "jump":
            # Convert jump command to specific actions
            actions.append({
                "type": "jump",
                "description": step.get("description", "Jump")
            })
            
        elif step.get("type") == "wait":
            # Convert wait command to specific actions
            duration = step.get("duration", 1.0)
            actions.append({
                "type": "wait",
                "duration": duration,
                "description": step.get("description", f"Wait for {duration} seconds")
            })
        
        # Handle any other action types by passing them through
        else:
            actions.append(step)
        
        # Advance to the next step
        self.current_plan_step += 1
        
        # Check if the plan is complete
        if self.current_plan_step >= len(self.current_plan):
            self.logger.info("Plan execution completed")
            # Clear the current plan to generate a new one next time
            self.current_plan = []
            self.current_plan_step = 0
        
        return actions
    
    def _generate_emergency_plan(self) -> List[Dict]:
        """
        Generate a plan for emergency situations
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        
        # Check if we have healing items
        if hasattr(self.current_state, "healing_items") and self.current_state.healing_items:
            # Use a healing item
            plan.append({
                "type": "use_item",
                "item": self.current_state.healing_items[0],
                "description": "Use healing item"
            })
        
        # Check if we're in combat and need to escape
        if getattr(self.current_state, "is_in_combat", False) and getattr(self.current_state, "health_percent", 100) < 15:
            # Try to escape combat
            if hasattr(self.current_state, "player_position") and hasattr(self.current_state, "nearby_entities"):
                # Find the farthest point from enemies
                enemies = [e for e in self.current_state.nearby_entities if e.get("hostile", False)]
                
                if enemies:
                    # Calculate average enemy position
                    avg_x = sum(e.get("position", [0, 0])[0] for e in enemies) / len(enemies)
                    avg_y = sum(e.get("position", [0, 0])[1] for e in enemies) / len(enemies)
                    
                    # Calculate a point in the opposite direction
                    player_pos = self.current_state.player_position
                    direction_x = player_pos[0] - avg_x
                    direction_y = player_pos[1] - avg_y
                    
                    # Normalize and scale the direction
                    magnitude = (direction_x**2 + direction_y**2)**0.5
                    if magnitude > 0:
                        direction_x = direction_x / magnitude * 50  # Run 50 units away
                        direction_y = direction_y / magnitude * 50
                    
                    # Create a flee position
                    flee_x = player_pos[0] + direction_x
                    flee_y = player_pos[1] + direction_y
                    flee_z = player_pos[2] if len(player_pos) > 2 else 0
                    
                    # Add flee movement to plan
                    flee_actions = self.navigation_manager.navigate_to(
                        self.current_state, 
                        (flee_x, flee_y, flee_z), 
                        getattr(self.current_state, "current_zone", None)
                    )
                    
                    plan.extend(flee_actions)
        
        return plan
    
    def _generate_loot_plan(self) -> List[Dict]:
        """
        Generate a plan for looting nearby corpses
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        
        # Find lootable entities
        lootable_entities = [e for e in getattr(self.current_state, "nearby_entities", []) 
                           if e.get("type") == "lootable"]
        
        for entity in lootable_entities:
            # Get entity position
            entity_pos = entity.get("position")
            if not entity_pos:
                continue
                
            # Check if the entity has a 3D position
            if len(entity_pos) < 3:
                entity_pos = (entity_pos[0], entity_pos[1], 0)
            
            # Add movement to the entity using advanced navigation
            move_actions = self.navigation_manager.navigate_to(
                self.current_state,
                entity_pos,
                getattr(self.current_state, "current_zone", None)
            )
            
            plan.extend(move_actions)
            
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
        vendor_entities = [e for e in getattr(self.current_state, "nearby_entities", []) 
                         if e.get("subtype") == "vendor"]
        
        if vendor_entities:
            vendor = vendor_entities[0]  # Choose the first vendor
            vendor_pos = vendor.get("position")
            
            if vendor_pos:
                # Add 3D coordinate if missing
                if len(vendor_pos) < 3:
                    vendor_pos = (vendor_pos[0], vendor_pos[1], 0)
                
                # Add movement to the vendor using advanced navigation
                move_actions = self.navigation_manager.navigate_to(
                    self.current_state,
                    vendor_pos,
                    getattr(self.current_state, "current_zone", None)
                )
                
                plan.extend(move_actions)
                
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
    
    def _generate_exploration_plan(self) -> List[Dict]:
        """
        Generate an intelligent exploration plan using terrain analysis
        
        Returns:
            List[Dict]: Plan steps
        """
        plan = []
        
        # Use terrain analysis to identify interesting areas
        terrain_analysis = self.navigation_manager.analyze_terrain(self.current_state)
        
        if terrain_analysis and hasattr(self.current_state, "player_position"):
            current_pos = self.current_state.player_position
            current_zone = getattr(self.current_state, "current_zone", None)
            
            # Try to find an interesting feature to explore
            exploration_target = None
            
            # Check for points of interest in the terrain analysis
            if terrain_analysis.get("jump_paths"):
                # Explore a jump path
                jump_path = terrain_analysis["jump_paths"][0]
                exploration_target = jump_path["end"]
                
                # Navigate to the jump start position
                move_actions = self.navigation_manager.navigate_to(
                    self.current_state,
                    jump_path["start"],
                    current_zone
                )
                
                plan.extend(move_actions)
                
                # Add jump action
                plan.append({
                    "type": "jump",
                    "description": "Jump to higher position"
                })
                
                # Navigate to the end position if needed
                end_actions = self.navigation_manager.navigate_to(
                    self.current_state,
                    jump_path["end"],
                    current_zone
                )
                
                plan.extend(end_actions)
                
                return plan
            
            # If no jump paths, try to find elevation changes or other features
            if not exploration_target and terrain_analysis.get("terrain_samples"):
                # Find a walkable high point to explore
                walkable_samples = [s for s in terrain_analysis["terrain_samples"] if s.get("walkable", False)]
                if walkable_samples:
                    # Sort by elevation (highest first)
                    walkable_samples.sort(key=lambda s: s["position"][2] if len(s["position"]) > 2 else 0, reverse=True)
                    exploration_target = walkable_samples[0]["position"]
            
            # If still no target, check for any resource nodes
            resource_nodes = [e for e in getattr(self.current_state, "nearby_entities", []) 
                            if e.get("type") == "resource_node"]
            
            if not exploration_target and resource_nodes:
                node = resource_nodes[0]
                node_pos = node.get("position")
                
                if node_pos:
                    # Add 3D coordinate if missing
                    if len(node_pos) < 3:
                        node_pos = (node_pos[0], node_pos[1], 0)
                    
                    exploration_target = node_pos
            
            # If still no target, use random exploration
            if not exploration_target:
                # Generate a random exploration target
                target_x = current_pos[0] + random.uniform(-100, 100)
                target_y = current_pos[1] + random.uniform(-100, 100)
                target_z = current_pos[2] if len(current_pos) > 2 else 0
                
                exploration_target = (target_x, target_y, target_z)
            
            # Navigate to the exploration target
            exploration_actions = self.navigation_manager.navigate_to(
                self.current_state, exploration_target, current_zone
            )
            
            plan.extend(exploration_actions)
        
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
    
    def _generate_fallback_actions(self, state: GameState) -> List[Dict]:
        """
        Generate fallback actions when behavior tree fails
        
        Args:
            state: Current game state
            
        Returns:
            List[Dict]: Fallback actions
        """
        # Try using the navigation integrator to get unstuck
        if hasattr(state, "player_position"):
            # Get current position
            current_pos = state.player_position
            current_zone = getattr(state, "current_zone", None)
            
            # Generate a nearby target to move to
            target_x = current_pos[0] + random.uniform(-20, 20)
            target_y = current_pos[1] + random.uniform(-20, 20)
            target_z = current_pos[2] if len(current_pos) > 2 else 0
            
            # Try to navigate there
            fallback_actions = self.navigation_manager.navigate_to(
                state, (target_x, target_y, target_z), current_zone
            )
            
            if fallback_actions:
                return fallback_actions
        
        # Simple random actions as fallback
        possible_actions = [
            {"type": "move", "position": (random.uniform(-20, 20), random.uniform(-20, 20), 0), "duration": 0.5},
            {"type": "jump"},
            {"type": "turn", "angle": random.uniform(-30, 30)},
            {"type": "wait", "duration": 0.5}
        ]
        
        # Choose 1-2 random actions
        num_actions = random.randint(1, 2)
        return random.sample(possible_actions, num_actions)
    
    # Public API for setting travel destinations
    def set_travel_destination(self, position: Tuple[float, float, float], zone: str = None) -> None:
        """
        Set a travel destination for the agent
        
        Args:
            position: Destination position (x, y, z)
            zone: Destination zone (defaults to current zone)
        """
        self.travel_destination = {
            "position": position,
            "zone": zone
        }
        self.logger.info(f"Set travel destination to {position} in zone {zone}")
    
    def navigate_to_dungeon(self, dungeon_name: str) -> None:
        """
        Navigate to a dungeon entrance
        
        Args:
            dungeon_name: Name of the dungeon
        """
        # Use the dungeon navigator to find the entrance
        dungeon_entrance = self.navigation_manager.navigation_manager.dungeon_navigator.get_dungeon_map(dungeon_name)
        
        if dungeon_entrance and dungeon_entrance.entrance_area and dungeon_entrance.entrance_position:
            # Set the travel destination to the dungeon entrance
            self.set_travel_destination(
                dungeon_entrance.entrance_position.to_tuple(),
                dungeon_entrance.entrance_area
            )
            self.logger.info(f"Set travel destination to dungeon: {dungeon_name}")
        else:
            self.logger.warning(f"Could not find entrance for dungeon: {dungeon_name}")
    
    def analyze_paths(self, state: GameState) -> Dict:
        """
        Analyze potential paths from current position
        
        Args:
            state: Current game state
            
        Returns:
            Dict: Analysis results with potential paths
        """
        # Get current zone and position
        current_zone = getattr(state, "current_zone", None)
        current_pos = getattr(state, "player_position", None)
        
        if not current_zone or not current_pos:
            return {"error": "Missing current zone or position"}
        
        # Get available flight paths
        flight_paths = self.navigation_manager.get_flight_paths(state)
        
        # Get dungeon information
        available_dungeons = []
        for dungeon_name in self.navigation_manager.navigation_manager.dungeon_navigator.dungeon_maps:
            dungeon_map = self.navigation_manager.navigation_manager.dungeon_navigator.get_dungeon_map(dungeon_name)
            if dungeon_map:
                available_dungeons.append({
                    "name": dungeon_map.name,
                    "type": dungeon_map.dungeon_type,
                    "entrance_zone": dungeon_map.entrance_area,
                    "boss_count": len(dungeon_map.boss_areas)
                })
        
        # Analyze terrain for interesting features
        terrain_analysis = self.navigation_manager.analyze_terrain(state)
        terrain_features = {
            "jump_paths": len(terrain_analysis.get("jump_paths", [])),
            "obstacles": len(terrain_analysis.get("obstacles", [])),
            "walkable_area": sum(1 for s in terrain_analysis.get("terrain_samples", []) if s.get("walkable", False))
        }
        
        # Compile the analysis
        analysis = {
            "current_zone": current_zone,
            "current_position": current_pos,
            "flight_paths": {
                "count": len(flight_paths),
                "destinations": [path.get("name") for path in flight_paths]
            },
            "dungeons": {
                "count": len(available_dungeons),
                "available": available_dungeons
            },
            "terrain": terrain_features,
            "navigation_mode": self.navigation_manager.current_navigation_mode
        }
        
        return analysis