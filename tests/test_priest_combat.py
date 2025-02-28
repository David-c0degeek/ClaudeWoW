"""
Tests for the Priest Combat Module
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.decision.combat.priest_combat_module import PriestCombatModule
from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge


class TestPriestCombatModule(unittest.TestCase):
    """Test cases for Priest Combat Module"""

    def setUp(self):
        """Set up the test environment"""
        # Mock config
        self.config = {
            "combat": {
                "priest": {
                    "rotation_delay": 0.5,
                    "health_threshold": 30,
                    "mana_threshold": 20
                }
            }
        }
        
        # Mock knowledge
        self.knowledge = MagicMock(spec=GameKnowledge)
        
        # Create priest combat module
        self.priest_module = PriestCombatModule(self.config, self.knowledge)
        
        # Create a basic game state
        self.state = GameState()
        self.state.player_class = "Priest"
        self.state.player_health = 100
        self.state.player_health_percent = 100
        self.state.player_mana = 100
        self.state.player_mana_percent = 100
        self.state.player_level = 60
        self.state.player_position = (100, 100, 30)
        self.state.player_buffs = []
        self.state.target = "Target1"
        self.state.target_health_percent = 100
        self.state.is_in_combat = True
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (110, 110, 30)
            }
        ]
    
    def test_initialization(self):
        """Test that the priest combat module initializes correctly"""
        self.assertIsInstance(self.priest_module, PriestCombatModule)
        self.assertEqual(self.priest_module.shield_target_id, None)
        self.assertEqual(self.priest_module.shadow_form_active, False)
        self.assertEqual(self.priest_module.inner_fire_active, False)
        self.assertEqual(self.priest_module.power_word_shield_cooldown, 0)
        self.assertEqual(self.priest_module.current_dot_targets, {})
    
    def test_detect_specialization(self):
        """Test specialization detection"""
        # Test shadow detection via shadowform
        self.state.player_buffs = [{"name": "Shadowform"}]
        self.priest_module._update_priest_state(self.state)
        spec = self.priest_module._detect_specialization(self.state)
        self.assertEqual(spec, "shadow")
        
        # Test action bar detection
        self.state.player_buffs = []
        self.state.action_bars = [
            {"name": "Mind Flay"},
            {"name": "Vampiric Embrace"},
            {"name": "Silence"}
        ]
        spec = self.priest_module._detect_specialization(self.state)
        self.assertEqual(spec, "shadow")
        
        # Test discipline spec
        self.state.action_bars = [
            {"name": "Power Word: Shield"},
            {"name": "Inner Focus"},
            {"name": "Power Infusion"}
        ]
        spec = self.priest_module._detect_specialization(self.state)
        self.assertEqual(spec, "discipline")
        
        # Test holy spec
        self.state.action_bars = [
            {"name": "Holy Nova"},
            {"name": "Circle of Healing"},
            {"name": "Prayer of Healing"}
        ]
        spec = self.priest_module._detect_specialization(self.state)
        self.assertEqual(spec, "holy")
        
        # Test fallback for no clear spec
        self.state.action_bars = [
            {"name": "Smite"},
            {"name": "Renew"}
        ]
        spec = self.priest_module._detect_specialization(self.state)
        self.assertEqual(spec, "generic")
    
    def test_shadow_rotation(self):
        """Test shadow priest rotation"""
        # Simulate shadowform not active
        self.state.player_buffs = []
        rotation = self.priest_module._get_shadow_rotation(self.state)
        
        # First ability should be Shadowform
        self.assertEqual(rotation[0]["spell"], "Shadowform")
        
        # Simulate shadowform active
        self.state.player_buffs = [{"name": "Shadowform"}]
        self.priest_module._update_priest_state(self.state)
        rotation = self.priest_module._get_shadow_rotation(self.state)
        
        # Should include DoTs in rotation
        spells = [item["spell"] for item in rotation]
        self.assertIn("Shadow Word: Pain", spells)
        self.assertIn("Mind Blast", spells)
        self.assertIn("Mind Flay", spells)
        
        # Emergency healing at low health
        self.state.player_health_percent = 20
        rotation = self.priest_module._get_shadow_rotation(self.state)
        self.assertEqual(rotation[0]["spell"], "Flash Heal")
        
        # Execute phase
        self.state.player_health_percent = 100
        self.state.target_health_percent = 15
        rotation = self.priest_module._get_shadow_rotation(self.state)
        spells = [item["spell"] for item in rotation]
        self.assertIn("Shadow Word: Death", spells)
    
    def test_discipline_rotation(self):
        """Test discipline priest rotation"""
        # Simulate Power Word: Shield not on cooldown
        self.priest_module.power_word_shield_cooldown = 0
        rotation = self.priest_module._get_discipline_rotation(self.state)
        
        # Should include basic disc abilities
        spells = [item["spell"] for item in rotation]
        self.assertIn("Power Word: Shield", spells)
        self.assertIn("Inner Fire", spells)
        
        # Test healing focus at low health
        self.state.player_health_percent = 50
        rotation = self.priest_module._get_discipline_rotation(self.state)
        self.assertIn({"type": "cast", "spell": "Flash Heal", "target": "player", "description": "Quick emergency self-heal"}, rotation)
        
        # Test mana recovery
        self.state.player_mana_percent = 15
        rotation = self.priest_module._get_discipline_rotation(self.state)
        spells = [item["spell"] for item in rotation]
        self.assertIn("Shadowfiend", spells)
        self.assertIn("Hymn of Hope", spells)
    
    def test_holy_rotation(self):
        """Test holy priest rotation"""
        # Set up group scenario
        self.state.group_members = [
            {"id": "Party1", "health_percent": 80, "role": "tank"},
            {"id": "Party2", "health_percent": 90}
        ]
        self.state.target_is_friendly = True
        
        rotation = self.priest_module._get_holy_rotation(self.state)
        
        # Should include basic holy abilities
        spells = [item["spell"] for item in rotation]
        self.assertIn("Inner Fire", spells)
        
        # Test emergency healing
        self.state.player_health_percent = 30
        rotation = self.priest_module._get_holy_rotation(self.state)
        self.assertIn({"type": "cast", "spell": "Guardian Spirit", "target": "player", "description": "Emergency protection and healing bonus"}, rotation)
        self.assertIn({"type": "cast", "spell": "Flash Heal", "target": "player", "description": "Fast emergency heal"}, rotation)
    
    def test_get_optimal_target(self):
        """Test target selection"""
        # Test hostile target selection
        target_info = self.priest_module.get_optimal_target(self.state, "shadow")
        self.assertEqual(target_info["id"], "Target1")
        
        # Test healing target selection for discipline/holy
        self.state.group_members = [
            {"id": "Party1", "health_percent": 60, "role": "tank"},
            {"id": "Party2", "health_percent": 90}
        ]
        
        target_info = self.priest_module.get_optimal_target(self.state, "discipline")
        self.assertEqual(target_info["id"], "Party1")  # Should prioritize the injured tank
    
    def test_get_optimal_position(self):
        """Test positioning logic"""
        # Test shadow priest positioning (should stay at range)
        position_info = self.priest_module.get_optimal_position(self.state, "shadow")
        self.assertEqual(position_info["description"], "Default position")  # Already at good range
        
        # Test positioning when too close for shadow
        self.state.player_position = (109, 109, 30)  # Very close to target
        position_info = self.priest_module.get_optimal_position(self.state, "shadow")
        self.assertEqual(position_info["description"], "Move away from target")
        
        # Test healing spec positioning when target is far
        self.state.player_position = (200, 200, 30)  # Far from target
        position_info = self.priest_module.get_optimal_position(self.state, "holy")
        self.assertEqual(position_info["description"], "Move closer to target")
    
    def test_get_supported_talent_builds(self):
        """Test talent build information"""
        builds = self.priest_module.get_supported_talent_builds()
        self.assertEqual(len(builds), 3)
        
        # Should have all three specs
        spec_names = [build["name"] for build in builds]
        self.assertIn("Discipline", spec_names)
        self.assertIn("Holy", spec_names)
        self.assertIn("Shadow", spec_names)
        
        # Check key abilities are listed
        shadow_build = next(b for b in builds if b["name"] == "Shadow")
        self.assertIn("Shadowform", shadow_build["key_abilities"])
        self.assertIn("Mind Blast", shadow_build["key_abilities"])


if __name__ == "__main__":
    unittest.main()