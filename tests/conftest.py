"""
PyTest configuration file for ClaudeWoW tests
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def mock_config():
    """Provide a standard mock configuration for tests"""
    return {
        "general": {
            "log_level": "DEBUG",
            "data_dir": "data",
            "debug_mode": True
        },
        "game": {
            "resolution": {
                "width": 1920,
                "height": 1080
            },
            "client_path": "C:/Path/To/Wow.exe",
            "key_bindings": {
                "forward": "w",
                "backward": "s",
                "left": "a",
                "right": "d",
                "jump": "space"
            }
        },
        "perception": {
            "screen_capture": {
                "fps": 30,
                "use_gpu": True
            },
            "ocr": {
                "enabled": True,
                "confidence_threshold": 0.7
            }
        },
        "decision": {
            "planning": {
                "max_plan_steps": 10,
                "replanning_frequency": 5
            },
            "combat": {
                "global": {
                    "health_threshold": 30,
                    "resource_threshold": 20,
                    "aoe_threshold": 3
                }
            },
            "navigation": {
                "path_smoothing": True,
                "obstacle_buffer": 2.0,
                "max_path_length": 1000
            }
        },
        "action": {
            "input_delay": 0.1,
            "key_press_duration": 0.05
        },
        "learning": {
            "deep_rl": {
                "state_dim": 100,
                "action_dim": 20,
                "hidden_dim": 256,
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "batch_size": 64
            },
            "imitation": {
                "state_dim": 100,
                "action_dim": 20,
                "hidden_dim": 256,
                "learning_rate": 0.0005,
                "batch_size": 32,
                "epochs": 10,
                "model_dir": "data/models/imitation"
            }
        },
        "social": {
            "chat": {
                "max_history": 50,
                "response_rate": 0.8,
                "harassment_threshold": 0.7
            },
            "reputation": {
                "decay_rate": 0.05,
                "decay_interval_days": 7,
                "max_reputation": 100,
                "min_reputation": -100
            }
        },
        "economic": {
            "market": {
                "price_history_days": 30,
                "update_frequency": 60,
                "arbitrage_threshold": 0.15
            },
            "inventory": {
                "value_threshold": 10,
                "bag_optimization": True
            }
        }
    }


@pytest.fixture
def mock_game_knowledge():
    """Provide a mock game knowledge instance"""
    knowledge = MagicMock()
    
    # Set up basic knowledge methods
    knowledge.get_ability_info.return_value = {
        "name": "Test Ability",
        "damage": 100,
        "cooldown": 10,
        "resource_cost": 20
    }
    
    knowledge.get_ability_cooldown.return_value = 10
    
    knowledge.get_available_abilities.return_value = [
        "Test Ability 1",
        "Test Ability 2",
        "Test Ability 3"
    ]
    
    knowledge.get_zone_info.return_value = {
        "name": "Test Zone",
        "level_range": [10, 20],
        "faction": "neutral"
    }
    
    knowledge.get_quest_objective_info.return_value = {
        "type": "kill",
        "target": "Test Monster",
        "count": 10
    }
    
    return knowledge


@pytest.fixture
def mock_game_state():
    """Provide a mock game state for testing"""
    from src.perception.screen_reader import GameState
    
    state = GameState()
    
    # Player info
    state.player_class = "warrior"
    state.player_level = 60
    state.player_health = 100
    state.player_health_max = 100
    state.player_mana = 100
    state.player_mana_max = 100
    state.player_rage = 50
    state.player_rage_max = 100
    state.player_position = (100, 100)
    state.player_buffs = []
    
    # Target info
    state.target = "Target1"
    state.target_health = 100
    state.target_level = 60
    state.is_in_combat = True
    
    # Environment info
    state.zone = "Test Zone"
    state.subzone = "Test Subzone"
    state.nearby_entities = [
        {
            "id": "Target1",
            "type": "mob",
            "reaction": "hostile",
            "level": 60,
            "health_percent": 100,
            "position": (110, 110)
        }
    ]
    
    return state


@pytest.fixture
def temp_data_dir(tmpdir):
    """Create a temporary data directory structure for tests"""
    # Create subdirectories
    models_dir = tmpdir.mkdir("models")
    models_dir.mkdir("combat")
    models_dir.mkdir("learning")
    models_dir.mkdir("social")
    
    data_dir = tmpdir.mkdir("data")
    data_dir.mkdir("recordings")
    data_dir.mkdir("logs")
    
    return tmpdir