"""
Decision Agent Module

This module handles high-level decision making and planning for the AI player.
"""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np

from src.perception.screen_reader import GameState, ScreenReader
from src.action.controller import Controller
from src.decision.behavior_tree import BehaviorTree, BehaviorStatus
from src.decision.planner import Planner
from src.decision.combat_manager import CombatManager
from src.decision.navigation_manager import NavigationManager
from src.decision.quest_manager import QuestManager
from src.knowledge.game_knowledge import GameKnowledge

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

        # Goals and plans
        self.current_goal = None
        self.current_plan = []
        self.current_plan_step = 0
        
        self.logger.info("Agent initialized")
    
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

            self.social_manager.update(game_state)
            
            # Run the behavior tree to determine high-level behavior
            behavior_status = self.behavior_tree.tick(game_state, self)
            
            if behavior_status == BehaviorStatus.FAILURE:
                self.logger.warning("Behavior tree execution failed")
                # Fallback to simple random actions for demonstration purposes
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

                    # Store the decided actions
                    
            self.last_actions = actions
            return actions
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {e}")
            self.logger.exception(e)
            return []
    
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