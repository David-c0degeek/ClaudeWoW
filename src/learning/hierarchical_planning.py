"""
Hierarchical Planning System

This module implements hierarchical task networks (HTN) planning and long-term
goal management for sophisticated AI planning capabilities.
"""

import logging
import time
import os
import json
import copy
from typing import Dict, List, Tuple, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid

# Setup module-level logger
logger = logging.getLogger("wow_ai.learning.hierarchical_planning")

class TaskStatus(Enum):
    """Status of a task in the hierarchical planner"""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    BLOCKED = auto()


@dataclass
class Condition:
    """A condition that must be satisfied for a task to be applicable"""
    name: str
    predicate: Callable[[Dict[str, Any]], bool]
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def check(self, state: Dict[str, Any]) -> bool:
        """Check if this condition is satisfied in the given state"""
        return self.predicate(state)


@dataclass
class Effect:
    """An effect that occurs when a task is completed"""
    name: str
    apply_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this effect to the given state"""
        return self.apply_func(state)


@dataclass
class Task:
    """A task in the hierarchical task network"""
    id: str
    name: str
    description: str
    preconditions: List[Condition] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    parent_id: Optional[str] = None
    priority: float = 0.5
    status: TaskStatus = TaskStatus.PENDING
    estimated_duration: float = 60.0  # seconds
    is_primitive: bool = False  # True for actions, False for compound tasks
    is_goal: bool = False       # True for top-level goals
    creation_time: float = field(default_factory=time.time)
    completion_time: Optional[float] = None
    
    def is_applicable(self, state: Dict[str, Any]) -> bool:
        """Check if this task is applicable in the given state"""
        return all(condition.check(state) for condition in self.preconditions)
    
    def apply_effects(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all effects of this task to the given state"""
        result = copy.deepcopy(state)
        for effect in self.effects:
            result = effect.apply(result)
        return result
    
    def is_completed(self) -> bool:
        """Check if this task is completed"""
        return self.status == TaskStatus.COMPLETED
    
    def set_completed(self) -> None:
        """Mark this task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completion_time = time.time()
        
    def reset(self) -> None:
        """Reset the status of this task and all subtasks"""
        self.status = TaskStatus.PENDING
        self.completion_time = None
        for subtask in self.subtasks:
            subtask.reset()
    
    def get_next_incomplete_primitive_task(self) -> Optional['Task']:
        """Get the next incomplete primitive task in this hierarchy"""
        # If this is a primitive task and it's not completed, return it
        if self.is_primitive and self.status != TaskStatus.COMPLETED:
            return self
        
        # Otherwise, look for an incomplete primitive subtask
        for subtask in self.subtasks:
            if subtask.status != TaskStatus.COMPLETED:
                result = subtask.get_next_incomplete_primitive_task()
                if result:
                    return result
        
        return None
    
    def get_all_primitive_tasks(self) -> List['Task']:
        """Get all primitive tasks in this hierarchy"""
        if self.is_primitive:
            return [self]
        
        result = []
        for subtask in self.subtasks:
            result.extend(subtask.get_all_primitive_tasks())
        return result


class HierarchicalTaskNetwork:
    """
    A hierarchical task network (HTN) for planning complex task sequences
    """
    
    def __init__(self):
        """Initialize the HTN"""
        self.tasks: Dict[str, Task] = {}
        self.task_methods: Dict[str, List[List[str]]] = {}  # Task name -> list of valid decompositions
        self.current_plan: List[str] = []  # IDs of tasks in current plan
        self.goals: List[str] = []  # IDs of current goal tasks
    
    def add_task(self, task: Task) -> str:
        """
        Add a task to the HTN
        
        Args:
            task: The task to add
            
        Returns:
            Task ID
        """
        if not task.id:
            task.id = str(uuid.uuid4())
        
        self.tasks[task.id] = task
        
        # If it's a goal, add to goals list
        if task.is_goal:
            self.goals.append(task.id)
        
        return task.id
    
    def add_task_method(self, task_name: str, subtask_names: List[str]) -> None:
        """
        Add a method (decomposition) for a task
        
        Args:
            task_name: Name of the task
            subtask_names: List of subtask names that decompose this task
        """
        if task_name not in self.task_methods:
            self.task_methods[task_name] = []
        
        self.task_methods[task_name].append(subtask_names)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            The task or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_tasks_by_name(self, name: str) -> List[Task]:
        """
        Get all tasks with the given name
        
        Args:
            name: Task name
            
        Returns:
            List of matching tasks
        """
        return [task for task in self.tasks.values() if task.name == name]
    
    def get_current_plan(self) -> List[Task]:
        """
        Get the current plan as a list of tasks
        
        Returns:
            List of tasks in the current plan
        """
        return [self.tasks[task_id] for task_id in self.current_plan if task_id in self.tasks]
    
    def get_active_goals(self) -> List[Task]:
        """
        Get all active goals
        
        Returns:
            List of active goal tasks
        """
        return [self.tasks[goal_id] for goal_id in self.goals 
                if goal_id in self.tasks and self.tasks[goal_id].status != TaskStatus.COMPLETED]
    
    def plan(self, state: Dict[str, Any]) -> bool:
        """
        Create a plan to achieve the active goals
        
        Args:
            state: Current world state
            
        Returns:
            True if planning succeeded, False otherwise
        """
        # Reset current plan
        self.current_plan = []
        
        # Get active goals
        active_goals = self.get_active_goals()
        
        if not active_goals:
            logger.info("No active goals to plan for")
            return True
        
        # Sort goals by priority
        active_goals.sort(key=lambda g: g.priority, reverse=True)
        
        # Try to plan for each goal in priority order
        for goal in active_goals:
            logger.info(f"Planning for goal: {goal.name}")
            plan_result = self._plan_for_task(goal, state)
            
            if plan_result:
                logger.info(f"Successfully planned for goal: {goal.name}")
                return True
        
        logger.warning("Failed to create a plan for any goal")
        return False
    
    def _plan_for_task(self, task: Task, state: Dict[str, Any]) -> bool:
        """
        Plan for a specific task
        
        Args:
            task: Task to plan for
            state: Current world state
            
        Returns:
            True if planning succeeded, False otherwise
        """
        # If task is not applicable, planning fails
        if not task.is_applicable(state):
            logger.debug(f"Task {task.name} is not applicable in current state")
            return False
        
        # If this is a primitive task, add it to the plan
        if task.is_primitive:
            self.current_plan.append(task.id)
            return True
        
        # If this is a compound task, decompose it
        if task.name in self.task_methods:
            # Try each method
            for method in self.task_methods[task.name]:
                method_succeeded = True
                
                # Create a temporary plan
                temp_plan = []
                
                # Current state for this method
                method_state = copy.deepcopy(state)
                
                # Try to plan for each subtask in the method
                for subtask_name in method:
                    # Find matching subtasks
                    matching_subtasks = [t for t in task.subtasks if t.name == subtask_name]
                    
                    if not matching_subtasks:
                        method_succeeded = False
                        break
                    
                    # Try each matching subtask
                    subtask_succeeded = False
                    for subtask in matching_subtasks:
                        if self._plan_for_task(subtask, method_state):
                            # Update the state with the effects of this subtask
                            method_state = subtask.apply_effects(method_state)
                            
                            # Add the subtask's plan to our temporary plan
                            temp_plan.extend(self.current_plan)
                            
                            # Clear the current plan for the next subtask
                            self.current_plan = []
                            
                            subtask_succeeded = True
                            break
                    
                    if not subtask_succeeded:
                        method_succeeded = False
                        break
                
                # If the method succeeded, update the current plan
                if method_succeeded:
                    self.current_plan = temp_plan
                    return True
            
            # If no method succeeded, planning fails
            return False
        
        # If the task has subtasks but no methods, try to plan for all subtasks in order
        if task.subtasks:
            all_succeeded = True
            temp_plan = []
            
            for subtask in task.subtasks:
                if self._plan_for_task(subtask, state):
                    # Update the state with the effects of this subtask
                    state = subtask.apply_effects(state)
                    
                    # Add the subtask's plan to our temporary plan
                    temp_plan.extend(self.current_plan)
                    
                    # Clear the current plan for the next subtask
                    self.current_plan = []
                else:
                    all_succeeded = False
                    break
            
            if all_succeeded:
                self.current_plan = temp_plan
                return True
            
            return False
        
        # If we get here, the task has no methods or subtasks
        logger.warning(f"Task {task.name} has no methods or subtasks")
        return False
    
    def execute_current_plan(self, state: Dict[str, Any]) -> bool:
        """
        Execute the current plan
        
        Args:
            state: Current world state
            
        Returns:
            True if execution succeeded, False otherwise
        """
        if not self.current_plan:
            logger.warning("No plan to execute")
            return False
        
        # Execute each task in the plan
        success = True
        for task_id in self.current_plan:
            task = self.tasks.get(task_id)
            
            if not task:
                logger.warning(f"Task {task_id} not found")
                success = False
                break
            
            if not task.is_applicable(state):
                logger.warning(f"Task {task.name} is not applicable in current state")
                success = False
                break
            
            # Execute the task
            logger.info(f"Executing task: {task.name}")
            
            # Update the state with the task's effects
            state = task.apply_effects(state)
            
            # Mark the task as completed
            task.set_completed()
        
        return success
    
    def get_next_executable_task(self, state: Dict[str, Any]) -> Optional[Task]:
        """
        Get the next executable task from the current plan
        
        Args:
            state: Current world state
            
        Returns:
            The next executable task or None if no tasks can be executed
        """
        for task_id in self.current_plan:
            task = self.tasks.get(task_id)
            
            if task and task.status != TaskStatus.COMPLETED and task.is_applicable(state):
                return task
        
        return None


class LongTermGoalManager:
    """
    Manager for long-term goals and goal decomposition
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the goal manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.htn = HierarchicalTaskNetwork()
        self.goal_library: Dict[str, Task] = {}  # Template goals
        self.active_goals: Dict[str, Task] = {}  # Currently active goals
        self.completed_goals: Dict[str, Task] = {}  # Completed goals
        
        # Goal management settings
        self.max_active_goals = config.get("planning", {}).get("max_active_goals", 3)
        self.goal_replan_interval = config.get("planning", {}).get("goal_replan_interval", 300)  # seconds
        self.last_replan_time = 0
        
        # Load goal library if available
        self._load_goal_library()
    
    def _load_goal_library(self) -> None:
        """Load the goal library from disk"""
        goals_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "goals"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(goals_dir, exist_ok=True)
        
        # Path for goal library
        library_path = os.path.join(goals_dir, "goal_library.json")
        
        if os.path.exists(library_path):
            try:
                with open(library_path, 'r') as f:
                    goal_data = json.load(f)
                
                # Convert the loaded data to Task objects
                for goal_id, goal_info in goal_data.items():
                    # Create a template goal
                    goal = Task(
                        id=goal_id,
                        name=goal_info["name"],
                        description=goal_info["description"],
                        priority=goal_info.get("priority", 0.5),
                        is_goal=True
                    )
                    
                    # Add to goal library
                    self.goal_library[goal_id] = goal
                
                logger.info(f"Loaded {len(self.goal_library)} goals from goal library")
            except Exception as e:
                logger.error(f"Failed to load goal library: {e}")
    
    def save_goal_library(self) -> None:
        """Save the goal library to disk"""
        goals_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "goals"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(goals_dir, exist_ok=True)
        
        # Path for goal library
        library_path = os.path.join(goals_dir, "goal_library.json")
        
        try:
            # Convert Task objects to serializable format
            goal_data = {}
            for goal_id, goal in self.goal_library.items():
                goal_data[goal_id] = {
                    "name": goal.name,
                    "description": goal.description,
                    "priority": goal.priority
                }
            
            with open(library_path, 'w') as f:
                json.dump(goal_data, f, indent=4)
            
            logger.info(f"Saved {len(self.goal_library)} goals to goal library")
        except Exception as e:
            logger.error(f"Failed to save goal library: {e}")
    
    def add_goal_to_library(self, goal: Task) -> str:
        """
        Add a goal to the goal library
        
        Args:
            goal: Goal to add
            
        Returns:
            Goal ID
        """
        if not goal.id:
            goal.id = str(uuid.uuid4())
        
        goal.is_goal = True
        self.goal_library[goal.id] = goal
        
        return goal.id
    
    def instantiate_goal(self, goal_id: str, parameters: Dict[str, Any] = None) -> Optional[str]:
        """
        Instantiate a goal from the library
        
        Args:
            goal_id: ID of the goal in the library
            parameters: Parameters for the goal
            
        Returns:
            ID of the instantiated goal, or None if failed
        """
        if goal_id not in self.goal_library:
            logger.warning(f"Goal {goal_id} not found in library")
            return None
        
        # Check if we're at the maximum number of active goals
        if len(self.active_goals) >= self.max_active_goals:
            logger.info("Maximum number of active goals reached, not adding new goal")
            return None
        
        # Create a copy of the template goal
        template = self.goal_library[goal_id]
        new_goal = copy.deepcopy(template)
        new_goal.id = str(uuid.uuid4())
        new_goal.creation_time = time.time()
        
        # Apply parameters if provided
        if parameters:
            # This would customize the goal based on parameters
            # For simplicity, we're just saving the parameters on the goal
            new_goal.parameters = parameters
        
        # Add to active goals
        self.active_goals[new_goal.id] = new_goal
        
        # Add to HTN
        self.htn.add_task(new_goal)
        
        logger.info(f"Instantiated goal: {new_goal.name} ({new_goal.id})")
        return new_goal.id
    
    def complete_goal(self, goal_id: str) -> bool:
        """
        Mark a goal as completed
        
        Args:
            goal_id: ID of the goal
            
        Returns:
            True if successful, False otherwise
        """
        if goal_id not in self.active_goals:
            logger.warning(f"Goal {goal_id} not found in active goals")
            return False
        
        goal = self.active_goals[goal_id]
        goal.set_completed()
        
        # Move from active to completed
        self.completed_goals[goal_id] = goal
        del self.active_goals[goal_id]
        
        logger.info(f"Completed goal: {goal.name} ({goal_id})")
        return True
    
    def abandon_goal(self, goal_id: str) -> bool:
        """
        Abandon an active goal
        
        Args:
            goal_id: ID of the goal
            
        Returns:
            True if successful, False otherwise
        """
        if goal_id not in self.active_goals:
            logger.warning(f"Goal {goal_id} not found in active goals")
            return False
        
        goal = self.active_goals[goal_id]
        goal.status = TaskStatus.FAILED
        
        # Just remove from active goals
        del self.active_goals[goal_id]
        
        logger.info(f"Abandoned goal: {goal.name} ({goal_id})")
        return True
    
    def check_goal_progress(self, state: Dict[str, Any]) -> None:
        """
        Check the progress of all active goals
        
        Args:
            state: Current world state
        """
        # Check if goals are completed based on their conditions
        goals_to_complete = []
        
        for goal_id, goal in self.active_goals.items():
            # Check if the goal is already marked as complete
            if goal.is_completed():
                goals_to_complete.append(goal_id)
                continue
                
            # Check if the goal's effects match the current state
            # This is a simplification - in a real system, we'd check
            # if the goal's success conditions are met
            goal_completed = True
            for effect in goal.effects:
                # Apply the effect to a copy of the state and check if it changes anything
                effect_state = effect.apply(copy.deepcopy(state))
                if effect_state != state:
                    goal_completed = False
                    break
            
            if goal_completed:
                goals_to_complete.append(goal_id)
        
        # Mark completed goals
        for goal_id in goals_to_complete:
            self.complete_goal(goal_id)
    
    def replan_if_needed(self, state: Dict[str, Any]) -> bool:
        """
        Replan if enough time has passed since the last replan
        
        Args:
            state: Current world state
            
        Returns:
            True if replanning occurred, False otherwise
        """
        current_time = time.time()
        
        if current_time - self.last_replan_time < self.goal_replan_interval:
            return False
        
        self.last_replan_time = current_time
        return self.replan(state)
    
    def replan(self, state: Dict[str, Any]) -> bool:
        """
        Create a new plan for the current goals
        
        Args:
            state: Current world state
            
        Returns:
            True if planning succeeded, False otherwise
        """
        # Update the HTN's goal list
        self.htn.goals = list(self.active_goals.keys())
        
        # Plan
        return self.htn.plan(state)
    
    def get_next_action(self, state: Dict[str, Any]) -> Optional[Task]:
        """
        Get the next action to take
        
        Args:
            state: Current world state
            
        Returns:
            The next action to take, or None if no actions are available
        """
        # Check if we need to replan
        if not self.htn.current_plan or self.replan_if_needed(state):
            self.replan(state)
        
        # Get the next executable task
        return self.htn.get_next_executable_task(state)
    
    def create_goal_decomposition(self, goal_id: str, subtasks: List[Task]) -> bool:
        """
        Create a decomposition for a goal
        
        Args:
            goal_id: ID of the goal
            subtasks: List of subtasks that decompose this goal
            
        Returns:
            True if successful, False otherwise
        """
        if goal_id not in self.goal_library and goal_id not in self.active_goals:
            logger.warning(f"Goal {goal_id} not found")
            return False
        
        # Get the goal
        goal = self.goal_library.get(goal_id, self.active_goals.get(goal_id))
        
        # Set subtasks
        goal.subtasks = subtasks
        
        # For each subtask, set its parent
        for subtask in subtasks:
            subtask.parent_id = goal.id
            
            # Add to HTN
            self.htn.add_task(subtask)
        
        # Add the decomposition method to the HTN
        self.htn.add_task_method(goal.name, [subtask.name for subtask in subtasks])
        
        logger.info(f"Created decomposition for goal {goal.name} with {len(subtasks)} subtasks")
        return True
    
    def learn_successful_decomposition(self, goal_name: str, successful_tasks: List[Task]) -> None:
        """
        Learn a successful decomposition for a goal
        
        Args:
            goal_name: Name of the goal
            successful_tasks: List of tasks that successfully achieved the goal
        """
        # Add the new decomposition
        self.htn.add_task_method(goal_name, [task.name for task in successful_tasks])
        
        logger.info(f"Learned new decomposition for {goal_name} with {len(successful_tasks)} tasks")
    
    def add_common_game_goals(self) -> None:
        """Add common WoW game goals to the library"""
        # Level Up Goal
        level_up = Task(
            id=str(uuid.uuid4()),
            name="level_up",
            description="Gain experience and level up the character",
            priority=0.8,
            is_goal=True
        )
        self.add_goal_to_library(level_up)
        
        # Complete Quest Goal
        complete_quest = Task(
            id=str(uuid.uuid4()),
            name="complete_quest",
            description="Complete a specific quest",
            priority=0.7,
            is_goal=True
        )
        self.add_goal_to_library(complete_quest)
        
        # Improve Gear Goal
        improve_gear = Task(
            id=str(uuid.uuid4()),
            name="improve_gear",
            description="Find better equipment for the character",
            priority=0.6,
            is_goal=True
        )
        self.add_goal_to_library(improve_gear)
        
        # Farm Resources Goal
        farm_resources = Task(
            id=str(uuid.uuid4()),
            name="farm_resources",
            description="Gather specific resources from the game world",
            priority=0.5,
            is_goal=True
        )
        self.add_goal_to_library(farm_resources)
        
        # Explore Zone Goal
        explore_zone = Task(
            id=str(uuid.uuid4()),
            name="explore_zone",
            description="Discover and map out a new zone",
            priority=0.4,
            is_goal=True
        )
        self.add_goal_to_library(explore_zone)
        
        logger.info("Added common game goals to library")


class HierarchicalPlanningManager:
    """
    Manager for hierarchical planning and long-term goal management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the planning manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.goal_manager = LongTermGoalManager(config)
        
        # Create data directories
        goals_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "goals"
        )
        os.makedirs(goals_dir, exist_ok=True)
        
        # Add common goals if the library is empty
        if not self.goal_manager.goal_library:
            self.goal_manager.add_common_game_goals()
            
        # Load task decompositions if available
        self._load_task_decompositions()
    
    def _load_task_decompositions(self) -> None:
        """Load task decompositions from disk"""
        decompositions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "goals", "task_decompositions.json"
        )
        
        if os.path.exists(decompositions_path):
            try:
                with open(decompositions_path, 'r') as f:
                    decompositions = json.load(f)
                
                # Add decompositions to the HTN
                for task_name, methods in decompositions.items():
                    for method in methods:
                        self.goal_manager.htn.add_task_method(task_name, method)
                
                logger.info(f"Loaded task decompositions for {len(decompositions)} tasks")
            except Exception as e:
                logger.error(f"Failed to load task decompositions: {e}")
    
    def save_task_decompositions(self) -> None:
        """Save task decompositions to disk"""
        decompositions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "goals", "task_decompositions.json"
        )
        
        try:
            with open(decompositions_path, 'w') as f:
                json.dump(self.goal_manager.htn.task_methods, f, indent=4)
            
            logger.info("Saved task decompositions")
        except Exception as e:
            logger.error(f"Failed to save task decompositions: {e}")
    
    def add_goal(self, name: str, description: str, priority: float = 0.5) -> str:
        """
        Add a new goal to the library
        
        Args:
            name: Goal name
            description: Goal description
            priority: Goal priority (0.0 to 1.0)
            
        Returns:
            Goal ID
        """
        goal = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            priority=priority,
            is_goal=True
        )
        
        return self.goal_manager.add_goal_to_library(goal)
    
    def activate_goal(self, goal_id: str, parameters: Dict[str, Any] = None) -> Optional[str]:
        """
        Activate a goal from the library
        
        Args:
            goal_id: ID of the goal in the library
            parameters: Parameters for the goal
            
        Returns:
            ID of the instantiated goal, or None if failed
        """
        return self.goal_manager.instantiate_goal(goal_id, parameters)
    
    def check_goal_status(self, state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Check the status of all goals
        
        Args:
            state: Current world state
            
        Returns:
            Dictionary with active, completed, and failed goals
        """
        # First, check goal progress
        self.goal_manager.check_goal_progress(state)
        
        # Prepare result
        result = {
            "active": [],
            "completed": [],
            "failed": []
        }
        
        # Add active goals
        for goal_id, goal in self.goal_manager.active_goals.items():
            result["active"].append({
                "id": goal_id,
                "name": goal.name,
                "description": goal.description,
                "priority": goal.priority,
                "status": goal.status.name
            })
        
        # Add completed goals
        for goal_id, goal in self.goal_manager.completed_goals.items():
            result["completed"].append({
                "id": goal_id,
                "name": goal.name,
                "description": goal.description,
                "priority": goal.priority,
                "completion_time": goal.completion_time
            })
        
        return result
    
    def create_task_hierarchy(self, parent_task_id: str, subtasks: List[Dict[str, Any]]) -> bool:
        """
        Create a task hierarchy
        
        Args:
            parent_task_id: ID of the parent task
            subtasks: List of subtask definitions
            
        Returns:
            True if successful, False otherwise
        """
        # Get the parent task
        parent_task = self.goal_manager.htn.get_task(parent_task_id)
        if not parent_task:
            logger.warning(f"Parent task {parent_task_id} not found")
            return False
        
        # Create subtasks
        task_objects = []
        for subtask_def in subtasks:
            subtask = Task(
                id=str(uuid.uuid4()),
                name=subtask_def["name"],
                description=subtask_def.get("description", ""),
                priority=subtask_def.get("priority", 0.5),
                is_primitive=subtask_def.get("is_primitive", False),
                parent_id=parent_task_id
            )
            
            task_objects.append(subtask)
        
        # Add decomposition
        return self.goal_manager.create_goal_decomposition(parent_task_id, task_objects)
    
    def get_next_action(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get the next action to take
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary describing the next action, or None if no actions are available
        """
        next_task = self.goal_manager.get_next_action(game_state)
        
        if not next_task:
            return None
        
        # Convert task to action
        action = {
            "task_id": next_task.id,
            "name": next_task.name,
            "description": next_task.description,
            "type": next_task.name  # Use task name as action type
        }
        
        return action
    
    def report_action_result(self, task_id: str, success: bool, state: Dict[str, Any]) -> None:
        """
        Report the result of an action
        
        Args:
            task_id: ID of the task/action
            success: Whether the action was successful
            state: Current world state
        """
        task = self.goal_manager.htn.get_task(task_id)
        
        if not task:
            logger.warning(f"Task {task_id} not found")
            return
        
        if success:
            # Mark the task as completed
            task.set_completed()
            
            # Apply the effects to the state
            task.apply_effects(state)
            
            logger.info(f"Task {task.name} completed successfully")
            
            # Check if the parent task is completed
            if task.parent_id:
                parent = self.goal_manager.htn.get_task(task.parent_id)
                if parent and all(subtask.is_completed() for subtask in parent.subtasks):
                    parent.set_completed()
                    
                    # If this was a goal, mark it as completed
                    if parent.is_goal and parent.id in self.goal_manager.active_goals:
                        self.goal_manager.complete_goal(parent.id)
        else:
            # Mark the task as failed
            task.status = TaskStatus.FAILED
            logger.warning(f"Task {task.name} failed")
            
            # Force replanning
            self.goal_manager.last_replan_time = 0
    
    def save_all(self) -> None:
        """Save all planning data"""
        self.goal_manager.save_goal_library()
        self.save_task_decompositions()
    
    def generate_complex_quest_plan(self, quest_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a complex plan for a quest
        
        Args:
            quest_details: Details about the quest
            
        Returns:
            List of plan steps
        """
        # Create a goal for the quest
        quest_goal = Task(
            id=str(uuid.uuid4()),
            name=f"complete_quest_{quest_details['id']}",
            description=f"Complete quest: {quest_details['name']}",
            priority=0.7,
            is_goal=True
        )
        
        # Add to goal library and activate it
        goal_id = self.goal_manager.add_goal_to_library(quest_goal)
        active_id = self.goal_manager.instantiate_goal(goal_id)
        
        if not active_id:
            return []
        
        # Create subtasks based on quest type and objectives
        subtasks = []
        
        # Common beginning tasks
        subtasks.append({
            "name": "travel_to_quest_giver",
            "description": f"Travel to {quest_details.get('quest_giver', 'quest giver')}",
            "is_primitive": True
        })
        
        subtasks.append({
            "name": "accept_quest",
            "description": f"Accept quest from {quest_details.get('quest_giver', 'quest giver')}",
            "is_primitive": True
        })
        
        # Objective-specific tasks
        for objective in quest_details.get("objectives", []):
            if objective["type"] == "kill":
                subtasks.append({
                    "name": "kill_mobs",
                    "description": f"Kill {objective.get('count', 1)} {objective.get('target', 'enemies')}",
                    "is_primitive": False
                })
                
                # Add decomposition for kill objective
                kill_subtasks = [
                    {
                        "name": "find_mobs",
                        "description": f"Locate {objective.get('target', 'enemies')}",
                        "is_primitive": True
                    },
                    {
                        "name": "engage_combat",
                        "description": "Engage in combat",
                        "is_primitive": True
                    },
                    {
                        "name": "defeat_enemies",
                        "description": "Defeat enemies in combat",
                        "is_primitive": True
                    },
                    {
                        "name": "loot_corpses",
                        "description": "Loot enemy corpses",
                        "is_primitive": True
                    }
                ]
                
                # Will add these subtasks later
                
            elif objective["type"] == "gather":
                subtasks.append({
                    "name": "gather_items",
                    "description": f"Gather {objective.get('count', 1)} {objective.get('item', 'items')}",
                    "is_primitive": False
                })
                
                # Add decomposition for gather objective
                gather_subtasks = [
                    {
                        "name": "find_items",
                        "description": f"Locate {objective.get('item', 'items')}",
                        "is_primitive": True
                    },
                    {
                        "name": "collect_items",
                        "description": f"Collect {objective.get('item', 'items')}",
                        "is_primitive": True
                    }
                ]
                
                # Will add these subtasks later
                
            elif objective["type"] == "escort":
                subtasks.append({
                    "name": "escort_npc",
                    "description": f"Escort {objective.get('target', 'NPC')} to {objective.get('destination', 'destination')}",
                    "is_primitive": True
                })
            
            elif objective["type"] == "explore":
                subtasks.append({
                    "name": "explore_location",
                    "description": f"Explore {objective.get('location', 'location')}",
                    "is_primitive": True
                })
        
        # Common ending tasks
        subtasks.append({
            "name": "return_to_quest_giver",
            "description": f"Return to {quest_details.get('quest_giver', 'quest giver')}",
            "is_primitive": True
        })
        
        subtasks.append({
            "name": "complete_quest",
            "description": f"Turn in quest to {quest_details.get('quest_giver', 'quest giver')}",
            "is_primitive": True
        })
        
        # Create the task hierarchy
        self.create_task_hierarchy(active_id, subtasks)
        
        # Create a plan
        self.goal_manager.replan({"player": {"level": quest_details.get("level_requirement", 1)}})
        
        # Convert the plan to a simple format
        plan = []
        for task_id in self.goal_manager.htn.current_plan:
            task = self.goal_manager.htn.get_task(task_id)
            if task:
                plan.append({
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "is_primitive": task.is_primitive
                })
        
        return plan