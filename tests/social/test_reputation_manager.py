"""
Tests for the Reputation Manager Module
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import with error handling for CI environments
try:
    from src.social.reputation_manager import ReputationManager, PlayerReputation, GuildReputation
    IMPORTS_SUCCEEDED = True
except ImportError:
    ReputationManager = MagicMock
    PlayerReputation = MagicMock
    GuildReputation = MagicMock
    IMPORTS_SUCCEEDED = False


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Reputation manager modules not available")
class TestReputationManager(unittest.TestCase):
    """Test cases for Reputation Manager module"""

    def setUp(self):
        """Set up the test environment"""
        # Mock config
        self.config = {
            "social": {
                "reputation": {
                    "decay_rate": 0.05,
                    "decay_interval_days": 7,
                    "max_reputation": 100,
                    "min_reputation": -100,
                    "data_path": "data/social/reputation.json"
                }
            }
        }
        
        # Mock knowledge
        self.knowledge = MagicMock()
        
        # Create reputation manager
        self.manager = ReputationManager(self.config, self.knowledge)
    
    def test_initialization(self):
        """Test that the manager initializes correctly"""
        self.assertIsInstance(self.manager, ReputationManager)
        self.assertEqual(self.manager.decay_rate, 0.05)
        self.assertEqual(self.manager.decay_interval_days, 7)
        self.assertEqual(self.manager.max_reputation, 100)
        self.assertEqual(self.manager.min_reputation, -100)
        self.assertEqual(self.manager.player_reputations, {})
        self.assertEqual(self.manager.guild_reputations, {})
    
    def test_update_player_reputation(self):
        """Test updating player reputation"""
        player_name = "TestPlayer"
        action = "helped_in_dungeon"
        
        # Set reputation impact for action
        self.manager.reputation_impacts = {
            "helped_in_dungeon": 10,
            "scammed_trade": -30
        }
        
        # Update reputation for positive action
        self.manager.update_player_reputation(player_name, action)
        
        # Check reputation was created and updated correctly
        self.assertIn(player_name, self.manager.player_reputations)
        player_rep = self.manager.player_reputations[player_name]
        self.assertEqual(player_rep.reputation_value, 10)
        
        # Test negative action
        self.manager.update_player_reputation(player_name, "scammed_trade")
        
        # Reputation should be reduced
        self.assertEqual(player_rep.reputation_value, -20)
        
        # Test clamping at min/max
        # Max first
        for _ in range(15):
            self.manager.update_player_reputation(player_name, "helped_in_dungeon")
        
        # Should be clamped at max
        self.assertEqual(player_rep.reputation_value, self.manager.max_reputation)
        
        # Then min
        for _ in range(10):
            self.manager.update_player_reputation(player_name, "scammed_trade")
        
        # Should be clamped at min
        self.assertEqual(player_rep.reputation_value, self.manager.min_reputation)
    
    def test_update_guild_reputation(self):
        """Test updating guild reputation"""
        guild_name = "TestGuild"
        action = "completed_guild_dungeon"
        
        # Set reputation impact for action
        self.manager.guild_reputation_impacts = {
            "completed_guild_dungeon": 5,
            "failed_guild_event": -10
        }
        
        # Update reputation for positive action
        self.manager.update_guild_reputation(guild_name, action)
        
        # Check reputation was created and updated correctly
        self.assertIn(guild_name, self.manager.guild_reputations)
        guild_rep = self.manager.guild_reputations[guild_name]
        self.assertEqual(guild_rep.reputation_value, 5)
        
        # Test negative action
        self.manager.update_guild_reputation(guild_name, "failed_guild_event")
        
        # Reputation should be reduced
        self.assertEqual(guild_rep.reputation_value, -5)
    
    def test_get_player_reputation(self):
        """Test retrieving player reputation"""
        # Create a player reputation
        player_name = "GetReputationPlayer"
        player_rep = PlayerReputation(player_name)
        player_rep.reputation_value = 25
        player_rep.interaction_history.append({
            "action": "helped_in_quest",
            "reputation_change": 25,
            "timestamp": datetime.now().isoformat()
        })
        self.manager.player_reputations[player_name] = player_rep
        
        # Get reputation
        reputation = self.manager.get_player_reputation(player_name)
        
        # Check returned reputation
        self.assertEqual(reputation["value"], 25)
        self.assertEqual(reputation["standing"], "friendly")  # Should map to friendly based on value
        self.assertEqual(len(reputation["recent_interactions"]), 1)
        
        # Test unknown player
        unknown_reputation = self.manager.get_player_reputation("UnknownPlayer")
        
        # Should return neutral standing for unknown players
        self.assertEqual(unknown_reputation["value"], 0)
        self.assertEqual(unknown_reputation["standing"], "neutral")
    
    def test_get_guild_reputation(self):
        """Test retrieving guild reputation"""
        # Create a guild reputation
        guild_name = "GetReputationGuild"
        guild_rep = GuildReputation(guild_name)
        guild_rep.reputation_value = 75
        guild_rep.interaction_history.append({
            "action": "completed_guild_raid",
            "reputation_change": 75,
            "timestamp": datetime.now().isoformat()
        })
        self.manager.guild_reputations[guild_name] = guild_rep
        
        # Get reputation
        reputation = self.manager.get_guild_reputation(guild_name)
        
        # Check returned reputation
        self.assertEqual(reputation["value"], 75)
        self.assertEqual(reputation["standing"], "exalted")  # Should map to exalted based on value
        self.assertEqual(len(reputation["recent_interactions"]), 1)
    
    def test_decay_reputations(self):
        """Test reputation decay over time"""
        # Create players with different last interaction times
        recent_player = "RecentPlayer"
        old_player = "OldPlayer"
        
        # Recent player (within decay interval)
        recent_rep = PlayerReputation(recent_player)
        recent_rep.reputation_value = 50
        recent_rep.last_interaction = datetime.now().isoformat()
        self.manager.player_reputations[recent_player] = recent_rep
        
        # Old player (outside decay interval)
        old_rep = PlayerReputation(old_player)
        old_rep.reputation_value = 50
        old_decay_date = datetime.now() - timedelta(days=self.manager.decay_interval_days + 1)
        old_rep.last_interaction = old_decay_date.isoformat()
        self.manager.player_reputations[old_player] = old_rep
        
        # Decay reputations
        self.manager.decay_reputations()
        
        # Recent player should not decay
        self.assertEqual(self.manager.player_reputations[recent_player].reputation_value, 50)
        
        # Old player should decay
        expected_value = 50 * (1 - self.manager.decay_rate)
        self.assertEqual(self.manager.player_reputations[old_player].reputation_value, expected_value)
    
    def test_save_load_reputation_data(self):
        """Test saving and loading reputation data"""
        # Create test reputation data
        player_name = "SaveLoadPlayer"
        player_rep = PlayerReputation(player_name)
        player_rep.reputation_value = 30
        self.manager.player_reputations[player_name] = player_rep
        
        guild_name = "SaveLoadGuild"
        guild_rep = GuildReputation(guild_name)
        guild_rep.reputation_value = 40
        self.manager.guild_reputations[guild_name] = guild_rep
        
        # Mock file operations
        mock_file = unittest.mock.mock_open()
        
        # Test save
        with patch('builtins.open', mock_file):
            with patch('json.dump') as mock_dump:
                self.manager.save_reputation_data()
                mock_dump.assert_called_once()
        
        # Test load
        mock_data = {
            "player_reputations": {
                "LoadedPlayer": {
                    "name": "LoadedPlayer",
                    "reputation_value": 60,
                    "last_interaction": datetime.now().isoformat(),
                    "interaction_history": []
                }
            },
            "guild_reputations": {
                "LoadedGuild": {
                    "name": "LoadedGuild",
                    "reputation_value": 70,
                    "last_interaction": datetime.now().isoformat(),
                    "interaction_history": []
                }
            }
        }
        
        with patch('builtins.open', mock_file):
            with patch('json.load', return_value=mock_data):
                self.manager.load_reputation_data()
                
                # Check data was loaded
                self.assertIn("LoadedPlayer", self.manager.player_reputations)
                self.assertEqual(
                    self.manager.player_reputations["LoadedPlayer"].reputation_value, 60)
                
                self.assertIn("LoadedGuild", self.manager.guild_reputations)
                self.assertEqual(
                    self.manager.guild_reputations["LoadedGuild"].reputation_value, 70)
    
    def test_analyze_player_interactions(self):
        """Test analyzing player interactions for patterns"""
        # Create player with interaction history
        player_name = "PatternPlayer"
        player_rep = PlayerReputation(player_name)
        
        # Add positive interactions
        for _ in range(5):
            player_rep.add_interaction("helped_in_dungeon", 10)
        
        # Add one negative interaction
        player_rep.add_interaction("failed_trade", -20)
        
        self.manager.player_reputations[player_name] = player_rep
        
        # Analyze interactions
        analysis = self.manager.analyze_player_interactions(player_name)
        
        # Check analysis results
        self.assertIn("total_interactions", analysis)
        self.assertEqual(analysis["total_interactions"], 6)
        
        self.assertIn("positive_ratio", analysis)
        self.assertAlmostEqual(analysis["positive_ratio"], 5/6, places=2)
        
        self.assertIn("common_actions", analysis)
        self.assertEqual(analysis["common_actions"][0]["action"], "helped_in_dungeon")
        
        self.assertIn("behavior_pattern", analysis)
    
    def test_assess_trade_risk(self):
        """Test assessing risk level for trades with a player"""
        # Create players with different reputation profiles
        good_player = "GoodTrader"
        bad_player = "BadTrader"
        unknown_player = "UnknownTrader"
        
        # Good trader with positive reputation
        good_rep = PlayerReputation(good_player)
        good_rep.reputation_value = 80
        for _ in range(10):
            good_rep.add_interaction("successful_trade", 8)
        self.manager.player_reputations[good_player] = good_rep
        
        # Bad trader with negative reputation
        bad_rep = PlayerReputation(bad_player)
        bad_rep.reputation_value = -50
        for _ in range(5):
            bad_rep.add_interaction("scammed_trade", -10)
        self.manager.player_reputations[bad_player] = bad_rep
        
        # Assess trade risks
        good_risk = self.manager.assess_trade_risk(good_player, 1000)
        bad_risk = self.manager.assess_trade_risk(bad_player, 1000)
        unknown_risk = self.manager.assess_trade_risk(unknown_player, 1000)
        
        # Check risk assessments
        self.assertEqual(good_risk["risk_level"], "low")
        self.assertEqual(bad_risk["risk_level"], "high")
        self.assertEqual(unknown_risk["risk_level"], "medium")
        
        # High value trades should increase risk
        high_value_unknown = self.manager.assess_trade_risk(unknown_player, 10000)
        self.assertEqual(high_value_unknown["risk_level"], "high")
    
    def test_get_reputation_standing(self):
        """Test mapping reputation values to standing labels"""
        # Test various reputation values
        self.assertEqual(self.manager.get_reputation_standing(-100), "hated")
        self.assertEqual(self.manager.get_reputation_standing(-70), "hostile")
        self.assertEqual(self.manager.get_reputation_standing(-30), "unfriendly")
        self.assertEqual(self.manager.get_reputation_standing(0), "neutral")
        self.assertEqual(self.manager.get_reputation_standing(20), "friendly")
        self.assertEqual(self.manager.get_reputation_standing(50), "honored")
        self.assertEqual(self.manager.get_reputation_standing(80), "revered")
        self.assertEqual(self.manager.get_reputation_standing(100), "exalted")


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Reputation manager modules not available")
class TestPlayerReputation(unittest.TestCase):
    """Test cases for the PlayerReputation class"""

    def setUp(self):
        """Set up the test environment"""
        self.player_name = "TestPlayer"
        self.reputation = PlayerReputation(self.player_name)
    
    def test_initialization(self):
        """Test that the player reputation initializes correctly"""
        self.assertIsInstance(self.reputation, PlayerReputation)
        self.assertEqual(self.reputation.name, self.player_name)
        self.assertEqual(self.reputation.reputation_value, 0)
        self.assertEqual(len(self.reputation.interaction_history), 0)
        self.assertIsNotNone(self.reputation.last_interaction)
    
    def test_add_interaction(self):
        """Test adding interaction records"""
        # Add an interaction
        self.reputation.add_interaction("helped_in_quest", 15)
        
        # Check reputation updated
        self.assertEqual(self.reputation.reputation_value, 15)
        
        # Check interaction recorded
        self.assertEqual(len(self.reputation.interaction_history), 1)
        self.assertEqual(self.reputation.interaction_history[0]["action"], "helped_in_quest")
        self.assertEqual(self.reputation.interaction_history[0]["reputation_change"], 15)
        
        # Add another interaction
        self.reputation.add_interaction("group_dungeon_run", 10)
        
        # Check cumulative reputation
        self.assertEqual(self.reputation.reputation_value, 25)
        self.assertEqual(len(self.reputation.interaction_history), 2)
    
    def test_get_recent_interactions(self):
        """Test retrieving recent interactions"""
        # Add several interactions
        self.reputation.add_interaction("interaction1", 5)
        self.reputation.add_interaction("interaction2", 10)
        self.reputation.add_interaction("interaction3", -5)
        self.reputation.add_interaction("interaction4", 15)
        self.reputation.add_interaction("interaction5", 20)
        
        # Get recent interactions (default limit is 5)
        recent = self.reputation.get_recent_interactions()
        self.assertEqual(len(recent), 5)
        
        # Get limited number of interactions
        limited = self.reputation.get_recent_interactions(3)
        self.assertEqual(len(limited), 3)
        
        # Most recent should be first
        self.assertEqual(limited[0]["action"], "interaction5")
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        # Add some data
        self.reputation.reputation_value = 45
        self.reputation.add_interaction("test_action", 45)
        
        # Convert to dict
        rep_dict = self.reputation.to_dict()
        
        # Check dictionary structure
        self.assertEqual(rep_dict["name"], self.player_name)
        self.assertEqual(rep_dict["reputation_value"], 45)
        self.assertEqual(len(rep_dict["interaction_history"]), 1)
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        # Create test data
        test_data = {
            "name": "DictPlayer",
            "reputation_value": 60,
            "last_interaction": datetime.now().isoformat(),
            "interaction_history": [
                {
                    "action": "test_action",
                    "reputation_change": 60,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        # Create from dict
        player_rep = PlayerReputation.from_dict(test_data)
        
        # Check object created correctly
        self.assertEqual(player_rep.name, "DictPlayer")
        self.assertEqual(player_rep.reputation_value, 60)
        self.assertEqual(len(player_rep.interaction_history), 1)


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Reputation manager modules not available")
class TestGuildReputation(unittest.TestCase):
    """Test cases for the GuildReputation class"""

    def setUp(self):
        """Set up the test environment"""
        self.guild_name = "TestGuild"
        self.reputation = GuildReputation(self.guild_name)
    
    def test_initialization(self):
        """Test that the guild reputation initializes correctly"""
        self.assertIsInstance(self.reputation, GuildReputation)
        self.assertEqual(self.reputation.name, self.guild_name)
        self.assertEqual(self.reputation.reputation_value, 0)
        self.assertEqual(len(self.reputation.interaction_history), 0)
        self.assertIsNotNone(self.reputation.last_interaction)
        self.assertEqual(self.reputation.member_count, 0)
        self.assertEqual(self.reputation.guild_level, 1)
    
    def test_update_guild_info(self):
        """Test updating guild information"""
        # Update guild info
        self.reputation.update_guild_info(member_count=25, guild_level=5)
        
        # Check updated info
        self.assertEqual(self.reputation.member_count, 25)
        self.assertEqual(self.reputation.guild_level, 5)
    
    def test_add_interaction(self):
        """Test adding guild interaction records"""
        # Add an interaction
        self.reputation.add_interaction("completed_guild_event", 20)
        
        # Check reputation updated
        self.assertEqual(self.reputation.reputation_value, 20)
        
        # Check interaction recorded
        self.assertEqual(len(self.reputation.interaction_history), 1)
        self.assertEqual(self.reputation.interaction_history[0]["action"], "completed_guild_event")
        
        # Add another interaction
        self.reputation.add_interaction("failed_guild_raid", -10)
        
        # Check cumulative reputation
        self.assertEqual(self.reputation.reputation_value, 10)
        self.assertEqual(len(self.reputation.interaction_history), 2)
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        # Add some data
        self.reputation.reputation_value = 70
        self.reputation.member_count = 30
        self.reputation.guild_level = 8
        self.reputation.add_interaction("test_guild_action", 70)
        
        # Convert to dict
        rep_dict = self.reputation.to_dict()
        
        # Check dictionary structure
        self.assertEqual(rep_dict["name"], self.guild_name)
        self.assertEqual(rep_dict["reputation_value"], 70)
        self.assertEqual(rep_dict["member_count"], 30)
        self.assertEqual(rep_dict["guild_level"], 8)
        self.assertEqual(len(rep_dict["interaction_history"]), 1)
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        # Create test data
        test_data = {
            "name": "DictGuild",
            "reputation_value": 85,
            "last_interaction": datetime.now().isoformat(),
            "member_count": 45,
            "guild_level": 10,
            "interaction_history": [
                {
                    "action": "test_guild_action",
                    "reputation_change": 85,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        # Create from dict
        guild_rep = GuildReputation.from_dict(test_data)
        
        # Check object created correctly
        self.assertEqual(guild_rep.name, "DictGuild")
        self.assertEqual(guild_rep.reputation_value, 85)
        self.assertEqual(guild_rep.member_count, 45)
        self.assertEqual(guild_rep.guild_level, 10)
        self.assertEqual(len(guild_rep.interaction_history), 1)


if __name__ == "__main__":
    unittest.main()