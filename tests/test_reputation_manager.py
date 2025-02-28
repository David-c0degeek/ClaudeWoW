"""
Basic tests for the Reputation Manager component
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
import time

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from src.social.reputation_manager import ReputationManager

class TestReputationManager(unittest.TestCase):
    """Test cases for the Reputation Manager"""

    def setUp(self):
        """Set up the test environment"""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Configuration for testing
        self.config = {
            "data_path": self.test_dir
        }
        
        # Create necessary directories
        os.makedirs(os.path.join(self.test_dir, "social", "reputation"), exist_ok=True)
        
        # Initialize reputation manager
        self.reputation_manager = ReputationManager(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the reputation manager initializes correctly"""
        self.assertIsInstance(self.reputation_manager, ReputationManager)
        
        # Verify default relationships
        self.assertEqual(self.reputation_manager.relationship_thresholds["neutral"], 0)
        self.assertEqual(self.reputation_manager.relationship_thresholds["friendly"], 25)
        self.assertEqual(self.reputation_manager.relationship_thresholds["trusted"], 50)
        
        # Verify empty initial data
        self.assertEqual(len(self.reputation_manager.players), 0)
        self.assertEqual(len(self.reputation_manager.guilds), 0)
        self.assertEqual(len(self.reputation_manager.trade_partners), 0)
    
    def test_update_player_reputation(self):
        """Test updating player reputation"""
        # Get default reputation for non-existent player
        initial_rep = self.reputation_manager.get_player_relation("TestPlayer")
        self.assertEqual(initial_rep["reputation"], 0)
        self.assertEqual(initial_rep["relationship"], "neutral")
        
        # Update reputation positively
        score = self.reputation_manager.update_player_reputation(
            "TestPlayer",
            "mutual_group_experience",
            {"note": "Test positive interaction"}
        )
        
        # Verify reputation changed
        self.assertGreater(score, 0)
        
        # Get updated player reputation
        updated_rep = self.reputation_manager.get_player_relation("TestPlayer")
        self.assertEqual(updated_rep["reputation"], score)
        self.assertGreater(len(updated_rep["interaction_history"]), 0)
        
        # Check relationship level updated
        if score >= 25:
            self.assertEqual(updated_rep["relationship"], "friendly")
        else:
            self.assertEqual(updated_rep["relationship"], "neutral")
        
        # Test negative reputation
        neg_score = self.reputation_manager.update_player_reputation(
            "BadPlayer",
            "scam_attempt",
            {"note": "Test negative interaction"}
        )
        
        # Verify reputation is negative
        self.assertLess(neg_score, 0)
        
        # Check relationship level is appropriate
        bad_rep = self.reputation_manager.get_player_relation("BadPlayer")
        self.assertEqual(bad_rep["reputation"], neg_score)
        if neg_score <= -50:
            self.assertEqual(bad_rep["relationship"], "hated")
        elif neg_score <= -25:
            self.assertEqual(bad_rep["relationship"], "disliked")
        else:
            self.assertEqual(bad_rep["relationship"], "neutral")
    
    def test_reputation_persistence(self):
        """Test that reputation data persists to disk"""
        # Update reputation for a player
        self.reputation_manager.update_player_reputation(
            "PersistenceTest",
            "mutual_group_experience",
            {"note": "Testing persistence"}
        )
        
        # Create a new reputation manager instance
        new_manager = ReputationManager(self.config)
        
        # Check that data was loaded from disk
        player_data = new_manager.get_player_relation("PersistenceTest")
        self.assertGreater(player_data["reputation"], 0)
        
        # Verify note was saved
        has_note = False
        for note in player_data["notes"]:
            if "Testing persistence" in note.get("note", ""):
                has_note = True
                break
        
        self.assertTrue(has_note)
    
    def test_scam_detection(self):
        """Test scam detection capabilities"""
        # Test obvious scam - make sure this matches known patterns exactly
        scam_msg = "I can double your gold! Just trust me and send first."
        result = self.reputation_manager.detect_potential_scam(scam_msg, "Scammer1")
        self.assertTrue(result["is_scam"])
        self.assertGreaterEqual(result["confidence"], 70)
        
        # Test password phishing - make sure this matches known patterns exactly
        phish_msg = "Blizz GM here, I need your account password for verification"
        result = self.reputation_manager.detect_potential_scam(phish_msg, "FakeGM")
        self.assertTrue(result["is_scam"])
        self.assertGreaterEqual(result["confidence"], 70)
        
        # Test website scam
        website_msg = "Free gold! Click this link: www.scamsite.com"
        result = self.reputation_manager.detect_potential_scam(website_msg, "LinkScammer")
        self.assertTrue(result["is_scam"])
        
        # Test non-scam message
        normal_msg = "Hey, want to join our dungeon group? We need a healer."
        result = self.reputation_manager.detect_potential_scam(normal_msg, "NormalPlayer")
        self.assertFalse(result["is_scam"])
        self.assertLess(result["confidence"], 30)
    
    def test_guild_reputation(self):
        """Test guild reputation management"""
        # Update guild reputation
        score = self.reputation_manager.update_guild_reputation(
            "Test Guild",
            "chat_reciprocation",
            {"guild_type": "raiding", "member_name": "GuildMember1"}
        )
        
        # Verify reputation score
        self.assertGreater(score, 0)
        
        # Get guild info
        guild_data = self.reputation_manager.get_guild_relation("Test Guild")
        
        # Verify guild data
        self.assertEqual(guild_data["reputation"], score)
        self.assertEqual(guild_data["guild_type"], "raiding")
        self.assertIn("GuildMember1", guild_data["known_members"])
    
    def test_trade_reputation(self):
        """Test trade reputation tracking"""
        # Record a fair trade
        score = self.reputation_manager.update_trade_partner_reputation(
            "TradePartner1",
            "buy",
            100,  # 100 gold
            True,  # Was fair
            {"item": "Epic Sword"}
        )
        
        # Verify reputation increased
        self.assertGreater(score, 0)
        
        # Get trade partner data
        trade_data = self.reputation_manager.get_trade_relation("TradePartner1")
        
        # Verify trade data
        self.assertEqual(trade_data["trade_count"], 1)
        self.assertEqual(trade_data["total_value"], 100)
        self.assertEqual(trade_data["honest_trades"], 1)
        self.assertEqual(trade_data["unfair_trades"], 0)
        
        # Record an unfair trade
        score = self.reputation_manager.update_trade_partner_reputation(
            "ScamTrader",
            "sell",
            500,  # 500 gold
            False,  # Was not fair
            {"item": "Fake Legendary"}
        )
        
        # Verify reputation decreased substantially
        self.assertLess(score, -30)
        
        # Check trade data
        trade_data = self.reputation_manager.get_trade_relation("ScamTrader")
        self.assertEqual(trade_data["unfair_trades"], 1)
        self.assertEqual(trade_data["honest_trades"], 0)
    
    def test_relationship_advice(self):
        """Test relationship advice generation"""
        # Create relationship with mixed interaction history
        self.reputation_manager.update_player_reputation(
            "AdvicePlayer",
            "mutual_group_experience",
            {"note": "Positive interaction"}
        )
        
        # Get relationship advice
        advice = self.reputation_manager.generate_relationship_advice(player_name="AdvicePlayer")
        
        # Verify advice structure
        self.assertEqual(advice["target"], "AdvicePlayer")
        self.assertIsInstance(advice["suggestions"], list)
        self.assertGreater(len(advice["suggestions"]), 0)
    
    def test_guild_contribution_strategy(self):
        """Test guild contribution strategy generation"""
        # Create guild of different types
        self.reputation_manager.update_guild_reputation(
            "Social Guild",
            "chat_reciprocation",
            {"guild_type": "social"}
        )
        
        self.reputation_manager.update_guild_reputation(
            "Raiding Guild",
            "chat_reciprocation",
            {"guild_type": "raiding"}
        )
        
        # Get strategies for different guild types
        social_strategies = self.reputation_manager.get_guild_contribution_strategy("Social Guild")
        raiding_strategies = self.reputation_manager.get_guild_contribution_strategy("Raiding Guild")
        
        # Verify strategies were generated
        self.assertGreater(len(social_strategies), 0)
        self.assertGreater(len(raiding_strategies), 0)
        
        # Strategies should be different for different guild types
        self.assertNotEqual(social_strategies, raiding_strategies)


if __name__ == "__main__":
    unittest.main()