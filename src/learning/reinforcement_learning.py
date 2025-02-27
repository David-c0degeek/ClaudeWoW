"""
Reinforcement Learning System

This module implements reinforcement learning algorithms for the AI agent
to learn optimal strategies through experience and rewards.
"""

import logging
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass

# Setup module-level logger
logger = logging.getLogger("wow_ai.learning.reinforcement_learning")

@dataclass
class Experience:
    """Represents a single learning experience for the agent"""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool


class ExperienceBuffer:
    """Buffer for storing and sampling experiences"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the experience buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
        
    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer
        
        Args:
            experience: The experience to add
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        return np.random.choice(self.buffer, min(batch_size, len(self.buffer)), replace=False).tolist()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str) -> None:
        """
        Save buffer to disk
        
        Args:
            path: Path to save the buffer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
        
    def load(self, path: str) -> bool:
        """
        Load buffer from disk
        
        Args:
            path: Path to load the buffer from
            
        Returns:
            Success status
        """
        try:
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
            return True
        except Exception as e:
            logger.error(f"Failed to load experience buffer: {e}")
            return False


class QLearning:
    """Q-Learning algorithm implementation"""
    
    def __init__(self, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.99, 
                 exploration_rate: float = 0.1,
                 exploration_decay: float = 0.995):
        """
        Initialize Q-Learning
        
        Args:
            learning_rate: Alpha - learning rate
            discount_factor: Gamma - future reward discount
            exploration_rate: Epsilon - exploration vs exploitation balance
            exploration_decay: Rate at which exploration decreases
        """
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.01
        
    def get_state_key(self, state: Dict[str, Any]) -> str:
        """
        Convert state dict to a string key
        
        Args:
            state: The state dictionary
            
        Returns:
            String representation of state
        """
        # Implement a way to convert complex state to a string key
        # This is a simplified version - you'd need a more robust implementation
        return str(sorted([(k, str(v)) for k, v in state.items()]))
    
    def get_q_value(self, state: Dict[str, Any], action: str) -> float:
        """
        Get Q value for state-action pair
        
        Args:
            state: The current state
            action: The action to evaluate
            
        Returns:
            Q value
        """
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        return self.q_table[state_key][action]
    
    def update_q_value(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]) -> None:
        """
        Update Q value for state-action pair
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Ensure state exists in q_table
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            
        # Find max Q for next state
        if next_state_key in self.q_table and self.q_table[next_state_key]:
            max_next_q = max(self.q_table[next_state_key].values())
        else:
            max_next_q = 0.0
            
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
    def choose_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        
        # Exploitation - choose best action based on Q values
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            return np.random.choice(available_actions)
        
        # Get Q values for all available actions
        q_values = {action: self.get_q_value(state, action) for action in available_actions}
        
        # Find action with max Q value
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        
        return np.random.choice(best_actions)
    
    def decay_exploration(self) -> None:
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str) -> None:
        """
        Save Q-table to disk
        
        Args:
            path: Path to save the Q-table
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, path: str) -> bool:
        """
        Load Q-table from disk
        
        Args:
            path: Path to load the Q-table from
            
        Returns:
            Success status
        """
        try:
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)
            return True
        except Exception as e:
            logger.error(f"Failed to load Q-table: {e}")
            return False


class ReinforcementLearningManager:
    """Manager class for reinforcement learning"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RL manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.experience_buffer = ExperienceBuffer(
            capacity=config.get("learning", {}).get("experience_buffer_size", 10000)
        )
        self.q_learning = QLearning(
            learning_rate=config.get("learning", {}).get("learning_rate", 0.1),
            discount_factor=config.get("learning", {}).get("discount_factor", 0.99),
            exploration_rate=config.get("learning", {}).get("exploration_rate", 0.1)
        )
        
        # Load saved models if available
        self._load_models()
        
    def _load_models(self) -> None:
        """Load saved reinforcement learning models"""
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Paths for saved models
        q_table_path = os.path.join(models_dir, "q_table.pkl")
        buffer_path = os.path.join(models_dir, "experience_buffer.pkl")
        
        # Try to load Q-table
        if os.path.exists(q_table_path):
            if self.q_learning.load(q_table_path):
                logger.info("Loaded Q-learning table")
        
        # Try to load experience buffer
        if os.path.exists(buffer_path):
            if self.experience_buffer.load(buffer_path):
                logger.info("Loaded experience buffer")
                
    def save_models(self) -> None:
        """Save reinforcement learning models"""
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Save Q-table
        q_table_path = os.path.join(models_dir, "q_table.pkl")
        self.q_learning.save(q_table_path)
        
        # Save experience buffer
        buffer_path = os.path.join(models_dir, "experience_buffer.pkl")
        self.experience_buffer.save(buffer_path)
        
        logger.info("Saved reinforcement learning models")
    
    def record_experience(self, state: Dict[str, Any], action: str, 
                         reward: float, next_state: Dict[str, Any], 
                         done: bool = False) -> None:
        """
        Record a learning experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether this experience ended an episode
        """
        experience = Experience(state, action, reward, next_state, done)
        self.experience_buffer.add(experience)
        
        # Update Q-values immediately
        self.q_learning.update_q_value(state, action, reward, next_state)
        
    def learn_from_batch(self, batch_size: int = 64) -> None:
        """
        Learn from a batch of experiences
        
        Args:
            batch_size: Number of experiences to learn from
        """
        if len(self.experience_buffer) < batch_size:
            return
        
        experiences = self.experience_buffer.sample(batch_size)
        
        for exp in experiences:
            self.q_learning.update_q_value(
                exp.state, exp.action, exp.reward, exp.next_state
            )
        
        # Decay exploration rate
        self.q_learning.decay_exploration()
        
    def choose_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Choose an action using current learning policy
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        return self.q_learning.choose_action(state, available_actions)