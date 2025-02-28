"""
Basic test for the Priest Combat Module structure
"""

import sys
import os
import unittest

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the required dependencies
import sys
from unittest.mock import MagicMock

class MockGameState:
    """Mock game state for testing"""
    def __init__(self):
        self.player_class = "Priest"
        self.player_health = 100
        self.player_health_percent = 100

# Mock the required modules
sys.modules['src.perception.screen_reader'] = MagicMock()
sys.modules['src.perception.screen_reader'].GameState = MockGameState
sys.modules['src.knowledge.game_knowledge'] = MagicMock()
sys.modules['src.knowledge.game_knowledge'].GameKnowledge = MagicMock

# Now import the module under test
from src.decision.combat.priest_combat_module import PriestCombatModule

class TestPriestCombatModuleBasic(unittest.TestCase):
    """Basic tests for Priest Combat Module"""

    def setUp(self):
        """Set up the test environment"""
        self.config = {"combat": {"priest": {}}}
        self.knowledge = MagicMock()
        self.priest_module = PriestCombatModule(self.config, self.knowledge)
    
    def test_module_exists(self):
        """Test that the module exists and is a PriestCombatModule"""
        self.assertIsNotNone(self.priest_module)
        self.assertIsInstance(self.priest_module, PriestCombatModule)
    
    def test_get_supported_talent_builds(self):
        """Test that the priest module returns talent builds"""
        builds = self.priest_module.get_supported_talent_builds()
        self.assertEqual(len(builds), 3)
        spec_names = [build["name"] for build in builds]
        self.assertIn("Discipline", spec_names)
        self.assertIn("Holy", spec_names)
        self.assertIn("Shadow", spec_names)

if __name__ == "__main__":
    unittest.main()