"""
Tests for the Imitation Learning Module
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import tempfile
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import module - handling import errors gracefully for CI environments without all dependencies
try:
    from src.learning.imitation_learning import ImitationLearningAgent, BehaviorCloner, GameplayDataset
    IMPORTS_SUCCEEDED = True
except ImportError:
    # Creating mock classes for testing environment without dependencies
    ImitationLearningAgent = MagicMock
    BehaviorCloner = MagicMock
    GameplayDataset = MagicMock
    IMPORTS_SUCCEEDED = False


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Imitation learning dependencies not available")
class TestImitationLearningAgent(unittest.TestCase):
    """Test cases for the Imitation Learning Agent class"""

    def setUp(self):
        """Set up the test environment"""
        self.config = {
            "learning": {
                "imitation": {
                    "state_dim": 100,
                    "action_dim": 20,
                    "hidden_dim": 256,
                    "learning_rate": 0.0005,
                    "batch_size": 32,
                    "epochs": 10,
                    "model_dir": "data/models/imitation"
                }
            }
        }
        
        # Create agent
        self.agent = ImitationLearningAgent(self.config)
    
    def test_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertIsInstance(self.agent, ImitationLearningAgent)
        self.assertEqual(self.agent.state_dim, 100)
        self.assertEqual(self.agent.action_dim, 20)
        self.assertEqual(self.agent.hidden_dim, 256)
        self.assertEqual(self.agent.learning_rate, 0.0005)
        self.assertEqual(self.agent.batch_size, 32)
        self.assertEqual(self.agent.epochs, 10)
    
    @patch('src.learning.imitation_learning.torch.tensor')
    @patch('src.learning.imitation_learning.torch.no_grad')
    def test_predict_action(self, mock_no_grad, mock_tensor):
        """Test action prediction"""
        # Mock model output
        self.agent.model = MagicMock()
        # Simulate action logits where action 5 has highest probability
        mock_output = np.zeros(20)
        mock_output[5] = 10.0
        self.agent.model.return_value = mock_output
        
        # Predict action
        state = np.zeros(100)
        action = self.agent.predict_action(state)
        
        # Should return the action with highest probability (5)
        self.assertEqual(action, 5)
    
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
        
        # Process state with a mock feature extractor
        with patch.object(self.agent, '_extract_features', return_value=np.zeros(100)):
            processed = self.agent.preprocess_state(raw_state)
            
            # Should be a numpy array of the correct shape
            self.assertIsInstance(processed, np.ndarray)
            self.assertEqual(processed.shape, (100,))
    
    def test_record_demonstration(self):
        """Test recording demonstration data"""
        # Create temp directory for demonstration data
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.agent.config["learning"]["imitation"]["demonstrations_dir"] = tmpdirname
            
            # Record a demonstration
            demo_data = {
                "states": [np.zeros(100) for _ in range(5)],
                "actions": [0, 1, 2, 3, 4],
                "metadata": {
                    "class": "warrior",
                    "level": 60,
                    "duration": 30
                }
            }
            
            self.agent.record_demonstration(demo_data, "test_demo")
            
            # Check that the file was created
            expected_file = os.path.join(tmpdirname, "test_demo.npz")
            self.assertTrue(os.path.exists(expected_file))
    
    @patch('src.learning.imitation_learning.np.load')
    def test_load_demonstration(self, mock_load):
        """Test loading demonstration data"""
        # Mock numpy load to return a dictionary-like object
        mock_data = {
            "states": np.zeros((5, 100)),
            "actions": np.array([0, 1, 2, 3, 4]),
            "metadata": np.array([json.dumps({
                "class": "warrior",
                "level": 60,
                "duration": 30
            })])
        }
        mock_load.return_value = mock_data
        
        # Load demonstration
        states, actions, metadata = self.agent.load_demonstration("test_demo.npz")
        
        # Check loaded data
        self.assertEqual(states.shape, (5, 100))
        self.assertEqual(len(actions), 5)
        self.assertEqual(metadata["class"], "warrior")
    
    @patch('src.learning.imitation_learning.torch.save')
    def test_save_model(self, mock_save):
        """Test model saving functionality"""
        # Create temp directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.agent.config["learning"]["imitation"]["model_dir"] = tmpdirname
            
            # Save model
            self.agent.save_model("test_model.pt")
            
            # Check if torch.save was called
            mock_save.assert_called_once()
    
    @patch('src.learning.imitation_learning.torch.load')
    def test_load_model(self, mock_load):
        """Test model loading functionality"""
        # Mock model
        self.agent.model = MagicMock()
        
        # Load model
        self.agent.load_model("test_model.pt")
        
        # Check if torch.load was called
        mock_load.assert_called_once()


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Imitation learning dependencies not available")
class TestBehaviorCloner(unittest.TestCase):
    """Test cases for the Behavior Cloner implementation"""

    def setUp(self):
        """Set up the test environment"""
        self.config = {
            "learning": {
                "behavior_cloning": {
                    "state_dim": 100,
                    "action_dim": 20,
                    "hidden_dim": 256,
                    "learning_rate": 0.0005,
                    "batch_size": 32,
                    "epochs": 10,
                    "model_dir": "data/models/behavior_cloning"
                }
            }
        }
        
        # Create cloner
        with patch('src.learning.imitation_learning.torch.nn.Sequential'):
            with patch('src.learning.imitation_learning.torch.optim.Adam'):
                self.cloner = BehaviorCloner(self.config)
    
    def test_initialization(self):
        """Test that the behavior cloner initializes correctly"""
        self.assertIsInstance(self.cloner, BehaviorCloner)
        self.assertEqual(self.cloner.state_dim, 100)
        self.assertEqual(self.cloner.action_dim, 20)
        self.assertEqual(self.cloner.hidden_dim, 256)
        self.assertEqual(self.cloner.learning_rate, 0.0005)
    
    @patch('src.learning.imitation_learning.DataLoader')
    @patch('src.learning.imitation_learning.GameplayDataset')
    @patch('src.learning.imitation_learning.torch.nn.CrossEntropyLoss')
    def test_train(self, mock_loss, mock_dataset, mock_dataloader):
        """Test training process"""
        # Mock training data
        states = np.zeros((100, 100))
        actions = np.zeros(100, dtype=np.int64)
        
        # Mock dataset and dataloader
        mock_dataset.return_value = "dataset"
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = [
            (np.zeros((32, 100)), np.zeros(32, dtype=np.int64)) 
            for _ in range(3)
        ]
        mock_dataloader.return_value = mock_loader
        
        # Mock loss and optimizer
        mock_loss_instance = MagicMock()
        mock_loss_instance.return_value = MagicMock()
        mock_loss.return_value = mock_loss_instance
        
        self.cloner.optimizer = MagicMock()
        self.cloner.model = MagicMock()
        
        # Train model
        self.cloner.train(states, actions, epochs=2)
        
        # Check if optimizer.step was called (indicating training occurred)
        self.assertGreater(self.cloner.optimizer.step.call_count, 0)
    
    @patch('src.learning.imitation_learning.torch.tensor')
    @patch('src.learning.imitation_learning.torch.no_grad')
    def test_predict(self, mock_no_grad, mock_tensor):
        """Test action prediction"""
        # Mock model output
        self.cloner.model = MagicMock()
        # Simulate action logits where action 3 has highest probability
        mock_output = np.zeros(20)
        mock_output[3] = 10.0
        self.cloner.model.return_value = mock_output
        
        # Predict action
        state = np.zeros(100)
        action = self.cloner.predict(state)
        
        # Should return the action with highest probability (3)
        self.assertEqual(action, 3)
    
    @patch('src.learning.imitation_learning.torch.save')
    def test_save(self, mock_save):
        """Test model saving functionality"""
        # Save model
        self.cloner.save("test_model.pt")
        
        # Check if torch.save was called
        mock_save.assert_called_once()
    
    @patch('src.learning.imitation_learning.torch.load')
    def test_load(self, mock_load):
        """Test model loading functionality"""
        # Mock model
        self.cloner.model = MagicMock()
        
        # Load model
        self.cloner.load("test_model.pt")
        
        # Check if torch.load was called
        mock_load.assert_called_once()


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Imitation learning dependencies not available")
class TestGameplayDataset(unittest.TestCase):
    """Test cases for the Gameplay Dataset class"""

    def setUp(self):
        """Set up the test environment"""
        # Create test data
        self.states = np.random.random((100, 50))
        self.actions = np.random.randint(0, 10, 100)
        
        # Create dataset
        self.dataset = GameplayDataset(self.states, self.actions)
    
    def test_initialization(self):
        """Test that the dataset initializes correctly"""
        self.assertIsInstance(self.dataset, GameplayDataset)
        self.assertEqual(len(self.dataset), 100)
    
    def test_getitem(self):
        """Test retrieving items from the dataset"""
        # Get first item
        state, action = self.dataset[0]
        
        # Check item types and shapes
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (50,))
        self.assertIsInstance(action, np.int64)
        
        # Check values match input data
        np.testing.assert_array_equal(state, self.states[0])
        self.assertEqual(action, self.actions[0])
    
    def test_len(self):
        """Test dataset length calculation"""
        self.assertEqual(len(self.dataset), 100)
        
        # Test with different length data
        states = np.random.random((50, 50))
        actions = np.random.randint(0, 10, 50)
        dataset = GameplayDataset(states, actions)
        self.assertEqual(len(dataset), 50)


if __name__ == "__main__":
    unittest.main()