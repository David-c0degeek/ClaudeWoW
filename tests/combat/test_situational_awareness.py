"""
Tests for the Combat Situational Awareness Module
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.decision.combat.situational_awareness import CombatSituationalAwareness
from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge


class TestCombatSituationalAwareness(unittest.TestCase):
    """Test cases for Combat Situational Awareness"""

    def setUp(self):
        """Set up the test environment"""
        # Mock config
        self.config = {
            "combat": {
                "situational_awareness": {
                    "aoe_threshold": 3,
                    "danger_threshold": 7
                }
            }
        }
        
        # Mock knowledge
        self.knowledge = MagicMock(spec=GameKnowledge)
        self.knowledge.get_spell_interrupt_priority.return_value = 0  # Default priority
        self.knowledge.get_spell_danger_rating.return_value = 0       # Default danger
        
        # Create situational awareness module
        self.awareness = CombatSituationalAwareness(self.config, self.knowledge)
        
        # Create a basic game state
        self.state = GameState()
        self.state.player_class = "warrior"
        self.state.player_health = 100
        self.state.player_health_max = 100
        self.state.player_level = 60
        self.state.player_position = (100, 100)
        self.state.player_role = "dps"
        self.state.is_in_combat = True
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "distance": 10,
                "position": (110, 110),
                "casting": None
            }
        ]
    
    def test_initialization(self):
        """Test that the situational awareness module initializes correctly"""
        self.assertIsInstance(self.awareness, CombatSituationalAwareness)
        self.assertEqual(self.awareness.aoe_opportunity, False)
        self.assertEqual(self.awareness.aoe_enemy_count, 0)
        self.assertEqual(self.awareness.interrupt_targets, [])
        self.assertEqual(self.awareness.cc_targets, [])
        self.assertEqual(self.awareness.in_pvp, False)
        self.assertEqual(self.awareness.danger_level, 0)
    
    def test_analyze_enemy_clusters(self):
        """Test enemy cluster analysis for AOE detection"""
        # Setup multiple enemies in a cluster
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (110, 110),
                "distance": 10
            },
            {
                "id": "Target2",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (112, 112),
                "distance": 12
            },
            {
                "id": "Target3",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (111, 109),
                "distance": 11
            },
            {
                "id": "FarTarget",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (150, 150),
                "distance": 50
            }
        ]
        
        # Analyze clusters
        self.awareness._analyze_enemy_clusters(self.state)
        
        # Should detect one cluster with 3 enemies
        self.assertEqual(len(self.awareness.enemy_clusters), 1)
        self.assertEqual(self.awareness.enemy_clusters[0]["size"], 3)
        self.assertEqual(self.awareness.aoe_enemy_count, 3)
        self.assertTrue(self.awareness.aoe_opportunity)
        
        # Test with spread out enemies
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (110, 110),
                "distance": 10
            },
            {
                "id": "Target2",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (130, 130),
                "distance": 30
            },
            {
                "id": "Target3",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (90, 90),
                "distance": 15
            }
        ]
        
        # Analyze clusters
        self.awareness._analyze_enemy_clusters(self.state)
        
        # Should detect no significant clusters
        self.assertEqual(self.awareness.aoe_enemy_count, 0)
        self.assertFalse(self.awareness.aoe_opportunity)
    
    def test_identify_interrupt_targets(self):
        """Test identification of targets that should be interrupted"""
        # Setup enemies casting interruptible spells
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (110, 110),
                "casting": {
                    "spell": "Healing Wave",
                    "remaining_time": 2.0
                }
            },
            {
                "id": "Target2",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (120, 120),
                "casting": {
                    "spell": "Fireball",
                    "remaining_time": 1.5
                }
            }
        ]
        
        # Setup knowledge to return priorities
        self.knowledge.get_spell_interrupt_priority.side_effect = lambda spell: 10 if spell == "Healing Wave" else 5
        self.knowledge.is_spell_cc_priority.return_value = False
        
        # Identify interrupt targets
        self.awareness._identify_interrupt_targets(self.state)
        
        # Should identify both targets for interruption
        self.assertEqual(len(self.awareness.interrupt_targets), 2)
        
        # First target should be Healing Wave (higher priority)
        self.assertEqual(self.awareness.interrupt_targets[0]["spell"], "Healing Wave")
        self.assertEqual(self.awareness.interrupt_targets[0]["priority"], 10)
        
        # Test CC prioritization
        self.knowledge.is_spell_cc_priority.side_effect = lambda spell: spell == "Healing Wave"
        
        # Identify CC targets
        self.awareness._identify_interrupt_targets(self.state)
        
        # Should identify Healing Wave caster for CC
        self.assertEqual(len(self.awareness.cc_targets), 1)
        self.assertEqual(self.awareness.cc_targets[0]["id"], "Target1")
    
    def test_detect_pvp_situation(self):
        """Test PvP situation detection"""
        # Setup PvP scenario with enemy players
        self.state.nearby_entities = [
            {
                "id": "EnemyPlayer1",
                "type": "player",
                "reaction": "hostile",
                "level": 60,
                "class": "mage",
                "health_percent": 100,
                "position": (110, 110)
            },
            {
                "id": "EnemyPlayer2",
                "type": "player",
                "reaction": "hostile",
                "level": 60,
                "class": "priest",
                "health_percent": 100,
                "position": (115, 115)
            },
            {
                "id": "FriendlyPlayer",
                "type": "player",
                "reaction": "friendly",
                "level": 60,
                "class": "warrior",
                "health_percent": 100,
                "position": (105, 105)
            }
        ]
        
        # Detect PvP situation
        self.awareness._detect_pvp_situation(self.state)
        
        # Should detect PvP situation with 2 enemy players
        self.assertTrue(self.awareness.in_pvp)
        self.assertEqual(len(self.awareness.pvp_enemies), 2)
        
        # Test with no enemy players
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
                "id": "FriendlyPlayer",
                "type": "player",
                "reaction": "friendly",
                "level": 60,
                "health_percent": 100,
                "position": (105, 105)
            }
        ]
        
        # Detect PvP situation
        self.awareness._detect_pvp_situation(self.state)
        
        # Should not detect PvP situation
        self.assertFalse(self.awareness.in_pvp)
        self.assertEqual(len(self.awareness.pvp_enemies), 0)
    
    def test_analyze_group_situation(self):
        """Test group composition and role analysis"""
        # Setup group scenario
        self.state.group_members = ["player", "Member1", "Member2", "Member3"]
        self.state.group_members_health = {
            "player": 90,
            "Member1": 60,
            "Member2": 100,
            "Member3": 40
        }
        self.state.Member1_class = "Warrior"
        self.state.Member1_role = "tank"
        self.state.Member1_target = "Target1"
        self.state.Member2_class = "Priest"
        self.state.Member2_role = "healer"
        self.state.Member3_class = "Mage"
        
        # Analyze group situation
        self.awareness._analyze_group_situation(self.state)
        
        # Should detect group and roles
        self.assertTrue(self.awareness.is_in_group)
        self.assertEqual(len(self.awareness.group_members), 3)  # Excluding player
        
        # Should identify tank's target
        self.assertEqual(self.awareness.tank_target, "Target1")
        
        # Test role determination
        tank_member = next(m for m in self.awareness.group_members if m["role"] == "tank")
        self.assertEqual(tank_member["id"], "Member1")
        
        healer_member = next(m for m in self.awareness.group_members if m["role"] == "healer")
        self.assertEqual(healer_member["id"], "Member2")
    
    def test_identify_dangerous_abilities(self):
        """Test identification of dangerous enemy abilities"""
        # Setup enemies with dangerous abilities
        self.state.nearby_entities = [
            {
                "id": "Target1",
                "type": "mob",
                "reaction": "hostile",
                "level": 60,
                "health_percent": 100,
                "position": (110, 110),
                "casting": {
                    "spell": "Pyroblast",
                    "remaining_time": 2.5
                }
            }
        ]
        
        # Setup ground effects
        self.state.ground_effects = [
            {
                "name": "Fire Zone",
                "position": (105, 105),
                "radius": 5.0,
                "remaining_time": 8.0
            }
        ]
        
        # Setup knowledge to return danger ratings
        self.knowledge.get_spell_danger_rating.side_effect = lambda spell: 8 if spell == "Pyroblast" else 2
        self.knowledge.get_ground_effect_danger.side_effect = lambda effect: 9 if effect == "Fire Zone" else 3
        
        # Identify dangerous abilities
        self.awareness._identify_dangerous_abilities(self.state)
        
        # Should identify both the cast and ground effect as dangerous
        self.assertEqual(len(self.awareness.active_enemy_abilities), 2)
        
        # Danger level should be set to the highest danger rating
        self.assertEqual(self.awareness.danger_level, 9)
        
        # Check specific abilities
        abilities = [ability["name"] for ability in self.awareness.active_enemy_abilities]
        self.assertIn("Pyroblast", abilities)
        self.assertIn("Fire Zone", abilities)
    
    def test_generate_tactical_suggestions(self):
        """Test generation of tactical suggestions"""
        # Setup test conditions
        self.awareness.aoe_opportunity = True
        self.awareness.aoe_enemy_count = 4
        self.awareness.interrupt_targets = [
            {"id": "Target1", "spell": "Healing Wave", "priority": 10}
        ]
        self.awareness.cc_targets = [
            {"id": "Target2", "reason": "Casting Polymorph", "priority": 8}
        ]
        self.awareness.in_pvp = True
        self.awareness.pvp_enemies = [
            {"id": "EnemyHealer", "class": "Priest", "level": 60}
        ]
        self.awareness.is_in_group = True
        self.awareness.group_role = "dps"
        self.awareness.tank_target = "MainTarget"
        self.awareness.danger_level = 8
        
        # Generate tactical suggestions
        self.awareness._generate_tactical_suggestions(self.state)
        
        # Should generate suggestions for all detected situations
        self.assertGreater(len(self.awareness.tactical_suggestions), 3)
        
        # Check for specific suggestion types
        suggestions = " ".join(self.awareness.tactical_suggestions)
        self.assertIn("AOE", suggestions)
        self.assertIn("Interrupt", suggestions)
        self.assertIn("DANGER", suggestions)
        self.assertIn("tank", suggestions.lower())
    
    def test_get_aoe_opportunities(self):
        """Test getting AOE opportunity information"""
        # Setup enemy clusters
        self.awareness.enemy_clusters = [
            {
                "center": (110, 110),
                "size": 4,
                "entities": [{"id": "Target1"}, {"id": "Target2"}, {"id": "Target3"}, {"id": "Target4"}]
            }
        ]
        
        # Get AOE opportunities
        opportunities = self.awareness.get_aoe_opportunities()
        
        # Should return the enemy clusters
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0]["size"], 4)
    
    def test_get_danger_assessment(self):
        """Test getting danger assessment"""
        # Setup dangerous abilities
        self.awareness.danger_level = 7
        self.awareness.active_enemy_abilities = [
            {"name": "Pyroblast", "caster": "Target1", "danger_rating": 7, "type": "cast"}
        ]
        
        # Get danger assessment
        assessment = self.awareness.get_danger_assessment()
        
        # Should include danger level and abilities
        self.assertEqual(assessment["danger_level"], 7)
        self.assertEqual(len(assessment["dangerous_abilities"]), 1)
        self.assertEqual(assessment["dangerous_abilities"][0]["name"], "Pyroblast")


if __name__ == "__main__":
    unittest.main()