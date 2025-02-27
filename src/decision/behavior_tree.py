"""
Behavior Tree Module

This module implements a behavior tree for AI decision making.
"""

import logging
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Callable
import uuid

# Behavior status enum
class BehaviorStatus(Enum):
    """Status of behavior tree node execution"""
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

# Node types
class NodeType(Enum):
    """Types of behavior tree nodes"""
    SEQUENCE = 1
    SELECTOR = 2
    CONDITION = 3
    ACTION = 4
    DECORATOR = 5
    PARALLEL = 6

class BehaviorNode:
    """Base class for behavior tree nodes"""
    
    def __init__(self, name: str, node_type: NodeType):
        """
        Initialize a behavior tree node
        
        Args:
            name: Node name
            node_type: Type of node
        """
        self.name = name
        self.node_type = node_type
        self.id = str(uuid.uuid4())
        self.children = []
        self.parent = None
    
    def add_child(self, child: 'BehaviorNode') -> None:
        """
        Add a child node
        
        Args:
            child: Child node to add
        """
        self.children.append(child)
        child.parent = self
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the node's behavior
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        raise NotImplementedError("Subclasses must implement tick()")

class SequenceNode(BehaviorNode):
    """
    Sequence node - succeeds if all children succeed, fails if any child fails
    """
    
    def __init__(self, name: str):
        """
        Initialize a sequence node
        
        Args:
            name: Node name
        """
        super().__init__(name, NodeType.SEQUENCE)
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the sequence node's behavior
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        for child in self.children:
            status = child.tick(state, agent)
            
            if status != BehaviorStatus.SUCCESS:
                return status
        
        return BehaviorStatus.SUCCESS

class SelectorNode(BehaviorNode):
    """
    Selector node - succeeds if any child succeeds, fails if all children fail
    """
    
    def __init__(self, name: str):
        """
        Initialize a selector node
        
        Args:
            name: Node name
        """
        super().__init__(name, NodeType.SELECTOR)
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the selector node's behavior
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        for child in self.children:
            status = child.tick(state, agent)
            
            if status != BehaviorStatus.FAILURE:
                return status
        
        return BehaviorStatus.FAILURE

class ConditionNode(BehaviorNode):
    """
    Condition node - evaluates a condition function
    """
    
    def __init__(self, name: str, condition_func: Callable[[Any, Any], bool]):
        """
        Initialize a condition node
        
        Args:
            name: Node name
            condition_func: Function that evaluates a condition (returns True/False)
        """
        super().__init__(name, NodeType.CONDITION)
        self.condition_func = condition_func
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the condition node's behavior
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        if self.condition_func(state, agent):
            return BehaviorStatus.SUCCESS
        else:
            return BehaviorStatus.FAILURE

class ActionNode(BehaviorNode):
    """
    Action node - executes an action function
    """
    
    def __init__(self, name: str, action_func: Callable[[Any, Any], BehaviorStatus]):
        """
        Initialize an action node
        
        Args:
            name: Node name
            action_func: Function that performs an action (returns BehaviorStatus)
        """
        super().__init__(name, NodeType.ACTION)
        self.action_func = action_func
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the action node's behavior
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        return self.action_func(state, agent)

class DecoratorNode(BehaviorNode):
    """
    Decorator node - modifies the result of its child
    """
    
    def __init__(self, name: str, decorator_func: Callable[[BehaviorStatus], BehaviorStatus]):
        """
        Initialize a decorator node
        
        Args:
            name: Node name
            decorator_func: Function that modifies a behavior status
        """
        super().__init__(name, NodeType.DECORATOR)
        self.decorator_func = decorator_func
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the decorator node's behavior
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        if not self.children:
            return BehaviorStatus.FAILURE
        
        child_status = self.children[0].tick(state, agent)
        return self.decorator_func(child_status)

class ParallelNode(BehaviorNode):
    """
    Parallel node - executes all children simultaneously
    """
    
    def __init__(self, name: str, success_threshold: int = None, failure_threshold: int = None):
        """
        Initialize a parallel node
        
        Args:
            name: Node name
            success_threshold: Number of successful children required for success
            failure_threshold: Number of failed children required for failure
        """
        super().__init__(name, NodeType.PARALLEL)
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the parallel node's behavior
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        successes = 0
        failures = 0
        
        for child in self.children:
            status = child.tick(state, agent)
            
            if status == BehaviorStatus.SUCCESS:
                successes += 1
            elif status == BehaviorStatus.FAILURE:
                failures += 1
        
        # Determine success/failure based on thresholds
        if self.success_threshold is not None and successes >= self.success_threshold:
            return BehaviorStatus.SUCCESS
        
        if self.failure_threshold is not None and failures >= self.failure_threshold:
            return BehaviorStatus.FAILURE
        
        # If neither threshold is met, return running
        return BehaviorStatus.RUNNING

class BehaviorTree:
    """
    Behavior tree implementation for game AI
    """
    
    def __init__(self):
        """Initialize an empty behavior tree"""
        self.logger = logging.getLogger("wow_ai.decision.behavior_tree")
        self.nodes = {}
        self.root = None
    
    def add_sequence(self, name: str, parent: str = None) -> str:
        """
        Add a sequence node to the tree
        
        Args:
            name: Node name
            parent: Parent node name
        
        Returns:
            str: Node name
        """
        node = SequenceNode(name)
        self.nodes[name] = node
        
        if parent:
            if parent in self.nodes:
                self.nodes[parent].add_child(node)
            else:
                self.logger.warning(f"Parent node {parent} not found")
        elif not self.root:
            self.root = node
        
        return name
    
    def add_selector(self, name: str, parent: str = None) -> str:
        """
        Add a selector node to the tree
        
        Args:
            name: Node name
            parent: Parent node name
        
        Returns:
            str: Node name
        """
        node = SelectorNode(name)
        self.nodes[name] = node
        
        if parent:
            if parent in self.nodes:
                self.nodes[parent].add_child(node)
            else:
                self.logger.warning(f"Parent node {parent} not found")
        elif not self.root:
            self.root = node
        
        return name
    
    def add_condition(self, name: str, parent: str, condition_func: Callable[[Any, Any], bool]) -> str:
        """
        Add a condition node to the tree
        
        Args:
            name: Node name
            parent: Parent node name
            condition_func: Condition function
        
        Returns:
            str: Node name
        """
        node = ConditionNode(name, condition_func)
        self.nodes[name] = node
        
        if parent in self.nodes:
            self.nodes[parent].add_child(node)
        else:
            self.logger.warning(f"Parent node {parent} not found")
        
        return name
    
    def add_action(self, name: str, parent: str, action_func: Callable[[Any, Any], BehaviorStatus]) -> str:
        """
        Add an action node to the tree
        
        Args:
            name: Node name
            parent: Parent node name
            action_func: Action function
        
        Returns:
            str: Node name
        """
        node = ActionNode(name, action_func)
        self.nodes[name] = node
        
        if parent in self.nodes:
            self.nodes[parent].add_child(node)
        else:
            self.logger.warning(f"Parent node {parent} not found")
        
        return name
    
    def add_decorator(self, name: str, parent: str, 
                     decorator_func: Callable[[BehaviorStatus], BehaviorStatus]) -> str:
        """
        Add a decorator node to the tree
        
        Args:
            name: Node name
            parent: Parent node name
            decorator_func: Decorator function
        
        Returns:
            str: Node name
        """
        node = DecoratorNode(name, decorator_func)
        self.nodes[name] = node
        
        if parent in self.nodes:
            self.nodes[parent].add_child(node)
        else:
            self.logger.warning(f"Parent node {parent} not found")
        
        return name
    
    def add_parallel(self, name: str, parent: str = None,
                   success_threshold: int = None, failure_threshold: int = None) -> str:
        """
        Add a parallel node to the tree
        
        Args:
            name: Node name
            parent: Parent node name
            success_threshold: Number of successful children required for success
            failure_threshold: Number of failed children required for failure
        
        Returns:
            str: Node name
        """
        node = ParallelNode(name, success_threshold, failure_threshold)
        self.nodes[name] = node
        
        if parent:
            if parent in self.nodes:
                self.nodes[parent].add_child(node)
            else:
                self.logger.warning(f"Parent node {parent} not found")
        elif not self.root:
            self.root = node
        
        return name
    
    def tick(self, state: Any, agent: Any) -> BehaviorStatus:
        """
        Execute the behavior tree
        
        Args:
            state: Current game state
            agent: Agent instance
        
        Returns:
            BehaviorStatus: Status of the behavior execution
        """
        if not self.root:
            self.logger.warning("Behavior tree has no root node")
            return BehaviorStatus.FAILURE
        
        try:
            return self.root.tick(state, agent)
        except Exception as e:
            self.logger.error(f"Error executing behavior tree: {e}")
            return BehaviorStatus.FAILURE
    
    def print_tree(self, node: BehaviorNode = None, indent: int = 0) -> None:
        """
        Print the behavior tree structure
        
        Args:
            node: Current node (None for root)
            indent: Current indentation level
        """
        if node is None:
            node = self.root
        
        if node is None:
            print("Empty tree")
            return
        
        print("  " * indent + f"- {node.name} ({node.node_type.name})")
        
        for child in node.children:
            self.print_tree(child, indent + 1)