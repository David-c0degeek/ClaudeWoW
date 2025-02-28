"""
Tests for the Combat Manager
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.decision.combat_manager import CombatManager
from src.decision.combat.base_combat_module import BaseCombatModule
from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge


class TestCombatManager(unittest.TestCase):
    """Test cases for Combat Manager"""

    def setUp(self):
        """Set up the test environment"""
        # Mock config
        self.config = {
            "combat": {
                "global": {
                    "health_threshold": 30,
                    "resource_threshold": 20,
                    "aoe_threshold": 3
                }
            }
        }
        
        # Mock knowledge
        self.knowledge = MagicMock(spec=GameKnowledge)
        
        # Create combat manager
        self.combat_manager = CombatManager(self.config, self.knowledge)
        
        # Create a basic game state
        self.state = GameState()
        self.state.player_class = "warrior"
        self.state.player_health = 100
        self.state.player_health_max = 100
        self.state.player_rage = 50
        self.state.player_rage_max = 100
        self.state.player_level = 60
        self.state.player_position = (100, 100)
        self.state.player_buffs = []
        self.state.target = "Target1"
        self.state.is_in_combat = True
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (110, 110)
            }
        ]
    
    def test_initialization(self):
        """Test that the combat manager initializes correctly"""
        self.assertIsInstance(self.combat_manager, CombatManager)
        self.assertEqual(self.combat_manager.config, self.config)
        self.assertEqual(self.combat_manager.knowledge, self.knowledge)
        self.assertEqual(self.combat_manager.combat_modules, {})
        self.assertIsNone(self.combat_manager.current_target)
    
    def test_get_combat_module_warrior(self):
        """Test retrieving the warrior combat module"""
        # Set state to warrior
        self.state.player_class = "warrior"
        
        # Get the module
        module = self.combat_manager._get_combat_module(self.state)
        
        # Verify the returned module
        self.assertIsNotNone(module)
        self.assertEqual(module.__class__.__name__, "WarriorCombatModule")
        
        # Test caching
        module2 = self.combat_manager._get_combat_module(self.state)
        self.assertIs(module, module2, "Should return the same cached instance")
    
    def test_get_combat_module_mage(self):
        """Test retrieving the mage combat module"""
        # Set state to mage
        self.state.player_class = "mage"
        
        # Get the module
        module = self.combat_manager._get_combat_module(self.state)
        
        # Verify the returned module
        self.assertIsNotNone(module)
        self.assertEqual(module.__class__.__name__, "MageCombatModule")
    
    def test_get_combat_module_priest(self):
        """Test retrieving the priest combat module"""
        # Set state to priest
        self.state.player_class = "priest"
        
        # Get the module
        module = self.combat_manager._get_combat_module(self.state)
        
        # Verify the returned module
        self.assertIsNotNone(module)
        self.assertEqual(module.__class__.__name__, "PriestCombatModule")
    
    def test_get_combat_module_unknown_class(self):
        """Test retrieving a module for an unknown class"""
        # Set state to an unknown class
        self.state.player_class = "unknown_class"
        
        # The manager should try to import dynamically and fall back to None
        module = self.combat_manager._get_combat_module(self.state)
        self.assertIsNone(module)
    
    def test_generate_combat_plan_existing_module(self):
        """Test generating a combat plan with a valid module"""
        # Create a mock combat module
        mock_module = MagicMock(spec=BaseCombatModule)
        mock_module.generate_combat_plan.return_value = [
            {"type": "cast", "spell": "Test Ability", "target": "Target1"}
        ]
        
        # Install the mock module
        self.combat_manager.combat_modules["warrior"] = mock_module
        
        # Generate a plan
        plan = self.combat_manager.generate_combat_plan(self.state)
        
        # Verify the generated plan
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["type"], "cast")
        self.assertEqual(plan[0]["spell"], "Test Ability")
        
        # Verify the mock was called
        mock_module.generate_combat_plan.assert_called_once_with(self.state)
    
    def test_generate_fallback_plan(self):
        """Test generating a fallback plan when no module is available"""
        # Ensure no module is available
        self.combat_manager.combat_modules = {}
        
        # Mock _get_combat_module to return None
        with patch.object(self.combat_manager, '_get_combat_module', return_value=None):
            # Generate a plan
            plan = self.combat_manager.generate_combat_plan(self.state)
            
            # Verify the fallback plan
            self.assertGreater(len(plan), 0)
            self.assertEqual(plan[0]["type"], "cast")
    
    def test_find_suitable_target(self):
        """Test the suitable target finding logic"""
        # Setup multiple targets
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (110, 110)
            },
            {
                "id": "Target2",
                "type": "mob",
                "reaction": "hostile",
                "level": 58,
                "health_percent": 100,
                "position": (105, 105)
            },
            {
                "id": "FriendlyNPC",
                "type": "npc",
                "reaction": "friendly",
                "level": 60,
                "position": (95, 95)
            }
        ]
        
        # Test target selection when not in combat
        self.state.is_in_combat = False
        self.state.target = None
        
        target = self.combat_manager._find_suitable_target(self.state)
        
        # Should select a hostile mob, not friendly NPC
        self.assertIsNotNone(target)
        self.assertIn(target["id"], ["Target1", "Target2"])
        self.assertEqual(target["reaction"], "hostile")
        
        # Mock quest objectives to test quest target prioritization
        self.state.active_quests = [{
            "title": "Test Quest",
            "objectives": [
                {"name": "Kill Target2", "current": 0, "total": 1}
            ]
        }]
        
        # Mock knowledge to return quest info
        self.knowledge.get_quest_objective_info.return_value = {
            "type": "kill",
            "target": "Target2"
        }
        
        target = self.combat_manager._find_suitable_target(self.state)
        
        # Should prioritize the quest target
        self.assertEqual(target["id"], "Target2")
    
    def test_get_supported_classes(self):
        """Test getting supported classes"""
        supported_classes = self.combat_manager.get_supported_classes()
        
        # Core classes should be supported
        self.assertIn("Warrior", supported_classes)
        self.assertIn("Mage", supported_classes)
        self.assertIn("Priest", supported_classes)
    
    def test_get_talent_builds_for_class(self):
        """Test getting talent builds for a class"""
        # Create a mock combat module
        mock_module = MagicMock(spec=BaseCombatModule)
        mock_module.get_supported_talent_builds.return_value = [
            {"name": "Test Build", "description": "A test build"}
        ]
        
        # Install the mock module
        self.combat_manager.combat_modules["warrior"] = mock_module
        
        # Get builds for the class
        builds = self.combat_manager.get_talent_builds_for_class("Warrior")
        
        # Verify the returned builds
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0]["name"], "Test Build")
        
        # Verify the mock was called
        mock_module.get_supported_talent_builds.assert_called_once()


if __name__ == "__main__":
    unittest.main()