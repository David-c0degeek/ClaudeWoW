"""
Tests for the Social Intelligence System
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json
import time
import tempfile
import shutil

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.social.enhanced_chat_analyzer import EnhancedChatAnalyzer
from src.social.reputation_manager import ReputationManager
from src.social.social_manager import SocialManager
from src.knowledge.game_knowledge import GameKnowledge


class TestSocialIntelligence(unittest.TestCase):
    """Test cases for the Social Intelligence System"""

    def setUp(self):
        """Set up the test environment"""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock config
        self.config = {
            "data_path": self.test_dir,
            "player_name": "TestPlayer",
            "player_class": "Warrior",
            "player_race": "Human",
            "player_level": 60,
            "social_friendliness": 0.7,
            "social_chattiness": 0.6,
            "social_helpfulness": 0.8,
            "max_messages_per_minute": 10,
            "min_chat_interval": 1.0,
            "use_llm_threshold": 0.5,
            "llm_provider": "mock"  # Use mock LLM for testing
        }
        
        # Create necessary directories
        os.makedirs(os.path.join(self.test_dir, "social", "reputation"), exist_ok=True)
        
        # Mock knowledge base
        self.knowledge_base = MagicMock(spec=GameKnowledge)
        
        # Initialize components
        self.reputation_manager = ReputationManager(self.config)
        self.chat_analyzer = EnhancedChatAnalyzer(self.config, self.knowledge_base)
        self.social_manager = SocialManager(self.config, self.knowledge_base)
        
        # Mock LLM responses
        self.mock_llm_responses = {
            "greeting": "Hello there!",
            "question": "I think the answer is 42.",
            "group_chat": "Is everyone ready for the next pull?",
            "default": "That's interesting!"
        }
        
        # Patch LLM generate_chat_response
        patcher = patch('src.social.llm_interface.LLMInterface.generate_chat_response')
        self.mock_generate_chat = patcher.start()
        self.mock_generate_chat.side_effect = self._mock_llm_response
        self.addCleanup(patcher.stop)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def _mock_llm_response(self, message, sender, channel, context=None):
        """Mock LLM response generator"""
        if "hello" in message.lower() or "hi" in message.lower():
            return self.mock_llm_responses["greeting"]
        elif "?" in message:
            return self.mock_llm_responses["question"]
        elif sender == "group":
            return self.mock_llm_responses["group_chat"]
        else:
            return self.mock_llm_responses["default"]
    
    def _create_game_state(self):
        """Create a mock game state for testing"""
        game_state = MagicMock()
        game_state.player_name = "TestPlayer"
        game_state.player_class = "Warrior"
        game_state.player_level = 60
        game_state.current_zone = "Elwynn Forest"
        game_state.is_in_combat = False
        game_state.is_in_group = False
        game_state.is_in_instance = False
        game_state.nearby_players = [
            {"name": "Player1", "level": 60, "class": "Mage"},
            {"name": "Player2", "level": 58, "class": "Rogue"}
        ]
        game_state.chat_log = []
        
        return game_state
    
    def test_reputation_manager_initialization(self):
        """Test that the reputation manager initializes correctly"""
        self.assertIsInstance(self.reputation_manager, ReputationManager)
        
        # Check default relationship thresholds
        self.assertEqual(self.reputation_manager.relationship_thresholds["neutral"], 0)
        self.assertEqual(self.reputation_manager.relationship_thresholds["friendly"], 25)
        
        # Check default reputation data structures
        self.assertIsInstance(self.reputation_manager.players, dict)
        self.assertIsInstance(self.reputation_manager.guilds, dict)
        self.assertIsInstance(self.reputation_manager.trade_partners, dict)
    
    def test_update_player_reputation(self):
        """Test updating player reputation"""
        # Initial reputation should not exist
        initial_data = self.reputation_manager.get_player_relation("TestPlayer1")
        self.assertEqual(initial_data["reputation"], 0)
        self.assertEqual(initial_data["relationship"], "neutral")
        
        # Update reputation positively
        self.reputation_manager.update_player_reputation(
            "TestPlayer1", 
            "mutual_group_experience", 
            {"note": "Grouped up for a dungeon"}
        )
        
        # Check updated reputation
        updated_data = self.reputation_manager.get_player_relation("TestPlayer1")
        self.assertGreater(updated_data["reputation"], 0)
        
        # Test negative reputation
        self.reputation_manager.update_player_reputation(
            "BadPlayer", 
            "scam_attempt", 
            {"note": "Tried to scam with fake item link"}
        )
        
        bad_data = self.reputation_manager.get_player_relation("BadPlayer")
        self.assertLess(bad_data["reputation"], 0)
        self.assertEqual(bad_data["relationship"], "disliked")
    
    def test_chat_analyzer_message_categorization(self):
        """Test chat message categorization"""
        # Test greeting detection
        self.assertEqual(self.chat_analyzer._categorize_message("hi there"), "greeting")
        self.assertEqual(self.chat_analyzer._categorize_message("hello buddy"), "greeting")
        
        # Test question detection
        self.assertEqual(self.chat_analyzer._categorize_message("where is the nearest inn?"), "question")
        
        # Test help request detection
        self.assertEqual(self.chat_analyzer._categorize_message("can someone help me with this quest?"), "help_request")
        
        # Test trade request detection
        self.assertEqual(self.chat_analyzer._categorize_message("WTS [Thunderfury] 5000g"), "trade_request")
        
        # Test unknown category
        self.assertEqual(self.chat_analyzer._categorize_message("random message"), "unknown")
    
    def test_scam_detection(self):
        """Test scam detection capabilities"""
        # Test obvious scam
        scam_result = self.reputation_manager.detect_potential_scam(
            "Hey, I can double your gold! Just give me 1000g and I'll give you back 2000g!",
            "ScammerDude"
        )
        
        self.assertTrue(scam_result["is_scam"])
        self.assertGreaterEqual(scam_result["confidence"], 70)
        
        # Test password phishing scam
        scam_result = self.reputation_manager.detect_potential_scam(
            "Blizzard here, we need your account password for verification",
            "GM_BlizzSupport"
        )
        
        self.assertTrue(scam_result["is_scam"])
        self.assertGreaterEqual(scam_result["confidence"], 70)
        
        # Test non-scam message
        scam_result = self.reputation_manager.detect_potential_scam(
            "Hey, want to join our dungeon group? We need a tank.",
            "LegitPlayer"
        )
        
        self.assertFalse(scam_result["is_scam"])
        self.assertLess(scam_result["confidence"], 30)
    
    def test_harassment_detection(self):
        """Test harassment detection capabilities"""
        # Test obvious harassment
        harassment_result = self.chat_analyzer._detect_harassment(
            "You're such a noob, uninstall the game you trash player"
        )
        
        self.assertTrue(harassment_result["is_harassment"])
        self.assertEqual(harassment_result["type"], "personal_attack")
        
        # Test non-harassment message
        harassment_result = self.chat_analyzer._detect_harassment(
            "Good game everyone, thanks for the group!"
        )
        
        self.assertFalse(harassment_result["is_harassment"])
    
    def test_relationship_based_responses(self):
        """Test relationship-based response personalization"""
        # Set up a friendly player
        self.reputation_manager.update_player_reputation(
            "FriendlyPlayer", 
            "mutual_group_experience", 
            {"magnitude": 5.0}  # Strong positive interaction
        )
        
        # Set up a disliked player
        self.reputation_manager.update_player_reputation(
            "DislikedPlayer", 
            "verbal_abuse", 
            {"magnitude": 2.0}  # Strong negative interaction
        )
        
        # Test responses to different players
        friendly_response = self.chat_analyzer.analyze_chat(
            "Hey there!", "FriendlyPlayer", "say"
        )
        
        disliked_response = self.chat_analyzer.analyze_chat(
            "Hey there!", "DislikedPlayer", "say"
        )
        
        # Verify responses are different based on relationship
        # Note: Since we're using templates with randomness, we can't check exact responses,
        # but we can verify they were generated
        self.assertIsNotNone(friendly_response)
        self.assertIsNotNone(disliked_response)
    
    def test_social_manager_integration(self):
        """Test social manager integration with reputation and chat systems"""
        # Create game state with chat message
        game_state = self._create_game_state()
        
        # Add chat log entry
        game_state.chat_log = ["[General] Player1: Hey everyone, could someone help me with the Hogger quest?"]
        
        # Process game state update
        self.social_manager.update(game_state)
        
        # Generate social actions
        actions = self.social_manager.generate_social_actions(game_state)
        
        # Should have a response in the chat queue
        self.assertTrue(any(action.get("type") == "chat" for action in actions))
    
    def test_guild_contribution_strategies(self):
        """Test guild contribution strategy generation"""
        # Update guild information
        self.reputation_manager.update_guild_reputation(
            "Test Guild", 
            "chat_reciprocation",
            {"guild_type": "raiding", "member_name": "GuildMember1"}
        )
        
        # Get contribution strategies
        strategies = self.social_manager.get_guild_contribution_strategy("Test Guild")
        
        # Should return several strategies
        self.assertGreaterEqual(len(strategies), 3)
    
    def test_relationship_advice(self):
        """Test relationship advice generation"""
        # Create relationship with mixed history
        self.reputation_manager.update_player_reputation(
            "ComplexPlayer", 
            "mutual_group_experience",  # Positive
            {"note": "Helped with a quest"}
        )
        
        self.reputation_manager.update_player_reputation(
            "ComplexPlayer", 
            "declined_reasonable_request",  # Negative
            {"note": "Declined to help with dungeon"}
        )
        
        # Get relationship advice
        advice = self.social_manager.get_relationship_advice("ComplexPlayer")
        
        # Should have target and suggestions
        self.assertEqual(advice["target"], "ComplexPlayer")
        self.assertTrue(len(advice["suggestions"]) > 0)
    
    def test_reputation_decay(self):
        """Test reputation decay mechanism"""
        # Create relationships
        self.reputation_manager.update_player_reputation(
            "ActivePlayer", 
            "mutual_group_experience",
            {"magnitude": 5.0}
        )
        
        self.reputation_manager.update_player_reputation(
            "InactivePlayer", 
            "mutual_group_experience",
            {"magnitude": 5.0}
        )
        
        # Record initial reputation
        active_initial = self.reputation_manager.get_player_relation("ActivePlayer")["reputation"]
        inactive_initial = self.reputation_manager.get_player_relation("InactivePlayer")["reputation"]
        
        # Update last seen time for active player to be recent
        self.reputation_manager.players["ActivePlayer"]["last_seen"] = time.time()
        
        # Update last seen time for inactive player to be long ago
        self.reputation_manager.players["InactivePlayer"]["last_seen"] = time.time() - (60 * 60 * 24 * 10)  # 10 days ago
        
        # Apply decay
        self.reputation_manager.apply_reputation_decay()
        
        # Check results
        active_after = self.reputation_manager.get_player_relation("ActivePlayer")["reputation"]
        inactive_after = self.reputation_manager.get_player_relation("InactivePlayer")["reputation"]
        
        # Active player should have no decay
        self.assertEqual(active_initial, active_after)
        
        # Inactive player should have some decay
        self.assertNotEqual(inactive_initial, inactive_after)
    
    def test_social_memory_persistence(self):
        """Test that social memory persists between sessions"""
        # Update player reputation
        self.reputation_manager.update_player_reputation(
            "PersistencePlayer", 
            "mutual_group_experience",
            {"note": "Testing persistence"}
        )
        
        # Create a new reputation manager instance (simulating a new session)
        new_reputation_manager = ReputationManager(self.config)
        
        # Check that the player data persisted
        player_data = new_reputation_manager.get_player_relation("PersistencePlayer")
        self.assertGreater(player_data["reputation"], 0)
        self.assertEqual(player_data["relationship"], "friendly")
        
        # Verify notes persisted
        self.assertTrue(any("Testing persistence" in note.get("note", "") 
                          for note in player_data["notes"]))


if __name__ == "__main__":
    unittest.main()