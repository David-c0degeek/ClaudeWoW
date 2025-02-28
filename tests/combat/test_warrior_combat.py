"""
Tests for the Warrior Combat Module
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.decision.combat.classes.warrior import WarriorCombatModule
from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge


class TestWarriorCombatModule(unittest.TestCase):
    """Test cases for Warrior Combat Module"""

    def setUp(self):
        """Set up the test environment"""
        # Mock config
        self.config = {
            "combat": {
                "warrior": {
                    "rotation_delay": 0.5,
                    "health_threshold": 30,
                    "rage_threshold": 20
                }
            }
        }
        
        # Mock knowledge
        self.knowledge = MagicMock(spec=GameKnowledge)
        self.knowledge.get_available_abilities.return_value = [
            "Battle Stance", "Defensive Stance", "Berserker Stance",
            "Heroic Strike", "Mortal Strike", "Bloodthirst", "Shield Slam",
            "Rend", "Thunder Clap", "Whirlwind", "Execute"
        ]
        
        # Create warrior combat module
        self.warrior_module = WarriorCombatModule(self.config, self.knowledge)
        
        # Create a basic game state
        self.state = GameState()
        self.state.player_class = "Warrior"
        self.state.player_health = 100
        self.state.player_health_max = 100
        self.state.player_rage = 50
        self.state.player_rage_max = 100
        self.state.player_level = 60
        self.state.player_position = (100, 100)
        self.state.player_buffs = []
        self.state.target = "Target1"
        self.state.target_health = 100
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
        """Test that the warrior combat module initializes correctly"""
        self.assertIsInstance(self.warrior_module, WarriorCombatModule)
        self.assertEqual(self.warrior_module.stance, "battle")
        self.assertEqual(self.warrior_module.has_executed_charge, False)
        self.assertEqual(self.warrior_module.current_resources["rage"], 0)
        self.assertEqual(self.warrior_module.max_resources["rage"], 100)
    
    def test_update_state(self):
        """Test state update function"""
        # Setup state with rage value
        self.state.player_rage = 75
        
        # Update module state
        self.warrior_module.update_state(self.state)
        
        # Check if state was properly updated
        self.assertEqual(self.warrior_module.current_resources["rage"], 75)
        self.assertEqual(self.warrior_module.current_target, "Target1")
        
        # Test stance detection
        self.state.player_buffs = ["Battle Stance"]
        self.warrior_module._update_stance(self.state)
        self.assertEqual(self.warrior_module.stance, "battle")
        
        self.state.player_buffs = ["Defensive Stance"]
        self.warrior_module._update_stance(self.state)
        self.assertEqual(self.warrior_module.stance, "defensive")
        
        self.state.player_buffs = ["Berserker Stance"]
        self.warrior_module._update_stance(self.state)
        self.assertEqual(self.warrior_module.stance, "berserker")
    
    def test_arms_rotation(self):
        """Test arms warrior rotation"""
        # Ensure we're in battle stance
        self.warrior_module.stance = "battle"
        
        # Get the arms rotation
        rotation = self.warrior_module._get_arms_rotation(self.state)
        
        # Verify core abilities are present in correct priority order
        abilities = [item["name"] for item in rotation]
        
        # Core arms abilities should be present
        self.assertIn("Mortal Strike", abilities)
        self.assertIn("Rend", abilities)
        self.assertIn("Heroic Strike", abilities)
        
        # Check rage-dump logic
        self.warrior_module.current_resources["rage"] = 80
        rotation = self.warrior_module._get_arms_rotation(self.state)
        high_rage_abilities = [item for item in rotation if item["name"] == "Heroic Strike"]
        self.assertTrue(any(ability["priority"] > 50 for ability in high_rage_abilities), 
                       "Heroic Strike should have higher priority with excess rage")
        
        # Check execute phase
        self.state.target_data = {"health_percent": 15}
        rotation = self.warrior_module._get_arms_rotation(self.state)
        execute_abilities = [item for item in rotation if item["name"] == "Execute"]
        self.assertTrue(any(ability["priority"] > 80 for ability in execute_abilities),
                       "Execute should have high priority in execute phase")
    
    def test_fury_rotation(self):
        """Test fury warrior rotation"""
        # Ensure we're in berserker stance
        self.warrior_module.stance = "berserker"
        
        # Get the fury rotation
        rotation = self.warrior_module._get_fury_rotation(self.state)
        
        # Verify core abilities are present
        abilities = [item["name"] for item in rotation]
        
        # Core fury abilities should be present
        self.assertIn("Bloodthirst", abilities)
        self.assertIn("Whirlwind", abilities)
        
        # Check AoE handling
        self.warrior_module.nearby_enemies_count = 4
        rotation = self.warrior_module._get_fury_rotation(self.state)
        aoe_abilities = [item for item in rotation if item["name"] in ["Whirlwind", "Cleave"]]
        self.assertTrue(any(ability["priority"] > 80 for ability in aoe_abilities),
                       "AoE abilities should have high priority with multiple targets")
    
    def test_protection_rotation(self):
        """Test protection warrior rotation"""
        # Ensure we're in defensive stance
        self.warrior_module.stance = "defensive"
        
        # Get the protection rotation
        rotation = self.warrior_module._get_protection_rotation(self.state)
        
        # Verify core abilities are present
        abilities = [item["name"] for item in rotation]
        
        # Core protection abilities should be present
        self.assertIn("Shield Slam", abilities)
        self.assertIn("Thunder Clap", abilities)
        
        # Check Shield Block uptime
        shield_block_abilities = [item for item in rotation if item["name"] == "Shield Block"]
        self.assertTrue(any(ability["priority"] > 80 for ability in shield_block_abilities),
                       "Shield Block should have high priority for protection")
    
    def test_get_optimal_target(self):
        """Test target selection logic"""
        # Setup multiple targets
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 50,
                "position": (110, 110)
            },
            {
                "id": "Target2",
                "type": "mob",
                "reaction": "hostile", 
                "level": 60,
                "health_percent": 20,
                "position": (105, 105)
            },
            {
                "id": "Target3",
                "type": "mob",
                "reaction": "hostile",
                "level": 62,
                "health_percent": 100,
                "position": (120, 120)
            }
        ]
        
        # Test DPS spec targeting (should prioritize lower health)
        self.warrior_module.config["warrior_spec"] = "arms"
        target = self.warrior_module.get_optimal_target(self.state)
        self.assertEqual(target["id"], "Target2")
        
        # Test protection spec targeting (should prioritize closest)
        self.warrior_module.config["warrior_spec"] = "protection"
        target = self.warrior_module.get_optimal_target(self.state)
        self.assertEqual(target["id"], "Target2")
        
        # Test quest target prioritization
        self.state.nearby_entities[2]["id"] = "Quest Monster"
        quest_targets = ["Quest Monster"]
        with patch.object(self.warrior_module, '_get_quest_targets', return_value=quest_targets):
            target = self.warrior_module.get_optimal_target(self.state)
            self.assertEqual(target["id"], "Quest Monster")
    
    def test_get_optimal_position(self):
        """Test positioning logic"""
        # Setup player and target positions
        self.warrior_module.player_position = (100, 100)
        self.warrior_module.target_data = {"position": (110, 110)}
        
        # Test melee DPS positioning (should be behind target)
        self.warrior_module.config["warrior_spec"] = "arms"
        position = self.warrior_module.get_optimal_position(self.state)
        
        # For melee specs, if already in range, may return None (no movement needed)
        if position is not None:
            # Should either be behind target or very close
            distance = ((position[0] - 110)**2 + (position[1] - 110)**2)**0.5
            self.assertTrue(distance < 5, "Arms warrior should position close to target")
        
        # Test tank positioning (should be in front of target)
        self.warrior_module.config["warrior_spec"] = "protection"
        position = self.warrior_module.get_optimal_position(self.state)
        
        # For tank, if already in position, may return None
        if position is not None:
            # Should be very close to target
            distance = ((position[0] - 110)**2 + (position[1] - 110)**2)**0.5
            self.assertTrue(distance < 3, "Protection warrior should position very close to target")
        
        # Test out-of-range positioning
        self.warrior_module.player_position = (200, 200)  # Far from target
        position = self.warrior_module.get_optimal_position(self.state)
        
        # Should always return a position when out of range
        self.assertIsNotNone(position)
    
    def test_get_defensive_abilities(self):
        """Test defensive ability selection"""
        # Test low health scenario
        self.warrior_module.current_resources["health"] = 15
        self.warrior_module.max_resources["health"] = 100
        
        defensive_abilities = self.warrior_module.get_defensive_abilities(self.state)
        ability_names = [ability["name"] for ability in defensive_abilities]
        
        # Should include emergency defensive cooldowns
        self.assertIn("Last Stand", ability_names)
        self.assertIn("Shield Wall", ability_names)
        
        # Test multiple enemies scenario
        self.warrior_module.current_resources["health"] = 80
        self.warrior_module.max_resources["health"] = 100
        self.warrior_module.nearby_enemies_count = 4
        
        defensive_abilities = self.warrior_module.get_defensive_abilities(self.state)
        ability_names = [ability["name"] for ability in defensive_abilities]
        
        # Should include AoE defensive abilities
        self.assertIn("Intimidating Shout", ability_names)
    
    def test_get_supported_talent_builds(self):
        """Test supported talent build information"""
        builds = self.warrior_module.get_supported_talent_builds()
        
        # Should have all three specs
        self.assertEqual(len(builds), 3)
        spec_names = [build["name"] for build in builds]
        self.assertTrue("Arms" in spec_names[0])
        self.assertTrue("Fury" in spec_names[1])
        self.assertTrue("Protection" in spec_names[2])
        
        # Check rotation information is present
        arms_build = builds[0]
        self.assertIn("rotation_priority", arms_build)
        self.assertGreater(len(arms_build["rotation_priority"]), 0)
        
        # Check key talents are listed
        key_talents = arms_build["key_talents"]
        self.assertIn("Mortal Strike", key_talents)


if __name__ == "__main__":
    unittest.main()