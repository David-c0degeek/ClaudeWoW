"""
Tests for the Deep Reinforcement Learning Module
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import module - handling import errors gracefully for CI environments without all dependencies
try:
    from src.learning.deep_reinforcement_learning import DeepRLAgent, PPOAgent, GameEnvironment
    IMPORTS_SUCCEEDED = True
except ImportError:
    # Creating mock classes for testing environment without dependencies
    DeepRLAgent = MagicMock
    PPOAgent = MagicMock
    GameEnvironment = MagicMock
    IMPORTS_SUCCEEDED = False


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Deep learning dependencies not available")
class TestDeepRLAgent(unittest.TestCase):
    """Test cases for the base Deep RL Agent class"""

    def setUp(self):
        """Set up the test environment"""
        self.config = {
            "learning": {
                "deep_rl": {
                    "state_dim": 50,
                    "action_dim": 10,
                    "hidden_dim": 128,
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "batch_size": 64
                }
            }
        }
        
        # Mock environment
        self.env = MagicMock()
        self.env.observation_space.shape = (50,)
        self.env.action_space.n = 10
        
        # Create agent
        self.agent = DeepRLAgent(self.config, self.env)
    
    def test_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertIsInstance(self.agent, DeepRLAgent)
        self.assertEqual(self.agent.state_dim, 50)
        self.assertEqual(self.agent.action_dim, 10)
        self.assertEqual(self.agent.hidden_dim, 128)
        self.assertEqual(self.agent.lr, 0.001)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.batch_size, 64)
    
    def test_preprocess_state(self):
        """Test state preprocessing"""
        # Create a raw state
        raw_state = {
            "player_health": 80,
            "player_mana": 60,
            "target_health": 70,
            "target_distance": 15,
            "abilities_on_cooldown": [True, False, True, False, False]
        }
        
        # Process state
        with patch.object(self.agent, '_extract_features', return_value=np.zeros(50)):
            processed = self.agent.preprocess_state(raw_state)
            
            # Should be a numpy array of the correct shape
            self.assertIsInstance(processed, np.ndarray)
            self.assertEqual(processed.shape, (50,))
    
    def test_remember(self):
        """Test experience memory storage"""
        # Add experiences
        for i in range(5):
            self.agent.remember(
                np.zeros(50),          # state
                i,                      # action
                np.random.random(),     # reward
                np.ones(50),           # next_state
                False                   # done
            )
        
        # Check memory size
        self.assertEqual(len(self.agent.memory), 5)
    
    def test_choose_action_exploration(self):
        """Test action selection during exploration"""
        # Force exploration
        self.agent.epsilon = 1.0
        
        # Choose action
        action = self.agent.choose_action(np.zeros(50))
        
        # Should return a valid action
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 10)
    
    @patch('src.learning.deep_reinforcement_learning.torch.nn.functional.softmax')
    def test_choose_action_exploitation(self, mock_softmax):
        """Test action selection during exploitation"""
        # Setup mock for model output
        self.agent.model = MagicMock()
        self.agent.model.return_value = MagicMock()
        
        # Mock softmax to return a specific distribution
        mock_probs = np.zeros(10)
        mock_probs[5] = 1.0  # Make action 5 the best
        mock_softmax.return_value = mock_probs
        
        # Force exploitation
        self.agent.epsilon = 0.0
        
        # Choose action
        with patch('src.learning.deep_reinforcement_learning.torch.no_grad'):
            with patch('src.learning.deep_reinforcement_learning.torch.tensor'):
                action = self.agent.choose_action(np.zeros(50))
                
                # Should choose the action with highest probability (5)
                self.assertEqual(action, 5)
    
    @patch('src.learning.deep_reinforcement_learning.torch.save')
    def test_save_model(self, mock_save):
        """Test model saving functionality"""
        # Save model
        self.agent.save_model("test_model.pt")
        
        # Check if torch.save was called
        mock_save.assert_called_once()
    
    @patch('src.learning.deep_reinforcement_learning.torch.load')
    def test_load_model(self, mock_load):
        """Test model loading functionality"""
        # Mock model
        self.agent.model = MagicMock()
        
        # Load model
        self.agent.load_model("test_model.pt")
        
        # Check if torch.load was called
        mock_load.assert_called_once()


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Deep learning dependencies not available")
class TestPPOAgent(unittest.TestCase):
    """Test cases for the PPO Agent implementation"""

    def setUp(self):
        """Set up the test environment"""
        self.config = {
            "learning": {
                "ppo": {
                    "state_dim": 100,
                    "action_dim": 15,
                    "hidden_dim": 256,
                    "learning_rate": 0.0003,
                    "gamma": 0.99,
                    "ppo_epochs": 10,
                    "clip_epsilon": 0.2,
                    "batch_size": 64
                }
            }
        }
        
        # Mock environment
        self.env = MagicMock()
        self.env.observation_space.shape = (100,)
        self.env.action_space.n = 15
        
        # Skip creating actual PyTorch models for unit tests
        with patch('src.learning.deep_reinforcement_learning.PPOActorCritic'):
            self.agent = PPOAgent(self.config, self.env)
    
    def test_initialization(self):
        """Test that the PPO agent initializes correctly"""
        self.assertIsInstance(self.agent, PPOAgent)
        self.assertEqual(self.agent.state_dim, 100)
        self.assertEqual(self.agent.action_dim, 15)
        self.assertEqual(self.agent.hidden_dim, 256)
        self.assertEqual(self.agent.lr, 0.0003)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.ppo_epochs, 10)
        self.assertEqual(self.agent.clip_epsilon, 0.2)
    
    def test_remember(self):
        """Test trajectory storage"""
        # Add experiences to trajectory
        for i in range(5):
            self.agent.remember(
                np.zeros(100),         # state
                i,                      # action
                np.random.random(),     # reward
                np.ones(100),          # next_state
                i == 4,                 # done (True on last iteration)
                np.random.random(),     # log_prob
                np.random.random()      # value
            )
        
        # Check trajectory size
        self.assertEqual(len(self.agent.states), 5)
        self.assertEqual(len(self.agent.actions), 5)
        self.assertEqual(len(self.agent.rewards), 5)
        self.assertEqual(len(self.agent.dones), 5)
        self.assertEqual(len(self.agent.log_probs), 5)
        self.assertEqual(len(self.agent.values), 5)
        
        # Last experience should be marked as done
        self.assertTrue(self.agent.dones[4])
    
    @patch('src.learning.deep_reinforcement_learning.torch.no_grad')
    def test_choose_action(self, mock_no_grad):
        """Test action selection"""
        # Mock policy and value outputs
        self.agent.policy = MagicMock()
        mock_dist = MagicMock()
        mock_dist.sample.return_value = 3  # Return action 3
        mock_dist.log_prob.return_value = -0.5  # Log probability
        self.agent.policy.return_value = (mock_dist, 0.75)  # (distribution, value)
        
        # Choose action
        with patch('src.learning.deep_reinforcement_learning.torch.tensor'):
            action, log_prob, value = self.agent.choose_action(np.zeros(100))
            
            # Should return the sampled action, log_prob, and value
            self.assertEqual(action, 3)
            self.assertEqual(log_prob, -0.5)
            self.assertEqual(value, 0.75)
    
    @patch('src.learning.deep_reinforcement_learning.torch.optim.Adam')
    @patch('src.learning.deep_reinforcement_learning.torch.tensor')
    def test_update(self, mock_tensor, mock_adam):
        """Test policy and value function updating"""
        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        self.agent.optimizer = mock_optimizer
        
        # Add some experiences to trajectory
        for i in range(10):
            self.agent.states.append(np.zeros(100))
            self.agent.actions.append(i % 5)
            self.agent.rewards.append(float(i))
            self.agent.dones.append(i == 9)  # Last one is done
            self.agent.log_probs.append(-0.5)
            self.agent.values.append(0.5)
        
        # Mock policy output for update
        self.agent.policy = MagicMock()
        mock_dist = MagicMock()
        mock_dist.log_prob.return_value = -0.6  # Different from stored to trigger update
        mock_dist.entropy.return_value = 1.0
        self.agent.policy.return_value = (mock_dist, 0.6)  # (distribution, value)
        
        # Update policy
        self.agent.update()
        
        # Optimizer should be called for updates
        self.assertGreater(mock_optimizer.step.call_count, 0)
        
        # Trajectory should be cleared after update
        self.assertEqual(len(self.agent.states), 0)
        self.assertEqual(len(self.agent.actions), 0)
        self.assertEqual(len(self.agent.rewards), 0)


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Deep learning dependencies not available")
class TestGameEnvironment(unittest.TestCase):
    """Test cases for the Game Environment wrapper"""

    def setUp(self):
        """Set up the test environment"""
        self.config = {
            "environment": {
                "observation_dim": 80,
                "action_dim": 20,
                "max_steps": 1000,
                "reward_config": {
                    "kill_reward": 10.0,
                    "damage_scale": 0.1,
                    "death_penalty": -5.0
                }
            }
        }
        
        # Mock game interface
        self.game_interface = MagicMock()
        
        # Create environment
        self.env = GameEnvironment(self.config, self.game_interface)
    
    def test_initialization(self):
        """Test that the environment initializes correctly"""
        self.assertIsInstance(self.env, GameEnvironment)
        self.assertEqual(self.env.observation_space.shape, (80,))
        self.assertEqual(self.env.action_space.n, 20)
        self.assertEqual(self.env.max_steps, 1000)
    
    def test_reset(self):
        """Test environment reset"""
        # Mock game state for reset
        mock_state = {
            "player_health": 100,
            "player_mana": 100,
            "target_health": 100,
            "in_combat": False
        }
        self.game_interface.get_state.return_value = mock_state
        
        # Reset environment
        observation = self.env.reset()
        
        # Should return a valid observation
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (80,))
        
        # Step counter should be reset
        self.assertEqual(self.env.steps, 0)
        
        # Initial state should be stored
        self.assertEqual(self.env.current_state, mock_state)
    
    def test_step(self):
        """Test environment stepping"""
        # Set initial state
        self.env.current_state = {
            "player_health": 100,
            "player_mana": 100,
            "target_health": 100,
            "in_combat": True
        }
        self.env.steps = 0
        
        # Mock action execution and new state
        self.game_interface.execute_action.return_value = None
        next_state = {
            "player_health": 90,      # Took some damage
            "player_mana": 80,        # Used some mana
            "target_health": 80,      # Dealt some damage
            "in_combat": True
        }
        self.game_interface.get_state.return_value = next_state
        
        # Take a step
        observation, reward, done, info = self.env.step(5)  # Action 5
        
        # Should return valid step results
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (80,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Step should be counted
        self.assertEqual(self.env.steps, 1)
        
        # Action should be executed
        self.game_interface.execute_action.assert_called_once_with(5)
        
        # Current state should be updated
        self.assertEqual(self.env.current_state, next_state)
        
        # Test done condition - player death
        next_state["player_health"] = 0
        self.game_interface.get_state.return_value = next_state
        _, _, done, _ = self.env.step(1)
        self.assertTrue(done)
        
        # Test done condition - target death
        next_state["player_health"] = 90
        next_state["target_health"] = 0
        self.game_interface.get_state.return_value = next_state
        _, _, done, _ = self.env.step(1)
        self.assertTrue(done)
        
        # Test done condition - max steps
        next_state["target_health"] = 50
        self.game_interface.get_state.return_value = next_state
        self.env.steps = self.env.max_steps - 1
        _, _, done, _ = self.env.step(1)
        self.assertTrue(done)
    
    def test_reward_calculation(self):
        """Test reward function calculation"""
        # Set initial state
        previous_state = {
            "player_health": 100,
            "player_mana": 100,
            "target_health": 100,
            "in_combat": True,
            "kills": 0
        }
        
        # Test damage dealt reward
        current_state = {
            "player_health": 100,
            "player_mana": 90,
            "target_health": 80,    # Dealt 20 damage
            "in_combat": True,
            "kills": 0
        }
        
        reward = self.env._calculate_reward(previous_state, current_state)
        self.assertGreater(reward, 0)  # Should get positive reward for damage
        
        # Test damage taken penalty
        current_state = {
            "player_health": 80,    # Took 20 damage
            "player_mana": 100,
            "target_health": 100,
            "in_combat": True,
            "kills": 0
        }
        
        reward = self.env._calculate_reward(previous_state, current_state)
        self.assertLess(reward, 0)  # Should get negative reward for taking damage
        
        # Test kill reward
        current_state = {
            "player_health": 90, 
            "player_mana": 80,
            "target_health": 0,     # Target died
            "in_combat": True,
            "kills": 1              # Got a kill
        }
        
        reward = self.env._calculate_reward(previous_state, current_state)
        self.assertGreaterEqual(reward, self.env.reward_config["kill_reward"])
        
        # Test death penalty
        current_state = {
            "player_health": 0,     # Player died
            "player_mana": 50,
            "target_health": 80,
            "in_combat": True,
            "kills": 0
        }
        
        reward = self.env._calculate_reward(previous_state, current_state)
        self.assertLessEqual(reward, self.env.reward_config["death_penalty"])


if __name__ == "__main__":
    unittest.main()