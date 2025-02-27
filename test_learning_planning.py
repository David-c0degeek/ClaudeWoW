"""
Test script for the learning and planning system
"""
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the necessary classes to avoid import issues
class GameState:
    """Mock GameState class for testing"""
    def __init__(self):
        self.player_class = ""
        self.level = 0
        self.health_percent = 100
        self.mana_percent = 100
        self.is_in_combat = False
        self.is_alive = True
        self.xp = 0
        self.coordinates = (0, 0)
        self.current_zone = ""
        self.is_in_group = False
        self.is_moving = False
        self.is_resting = False
        self.active_quests = []
        self.current_quest = None
        self.quest_progress = 0
        self.nearby_entities = []
        self.inventory = []
        self.usable_items = []
        self.target = None

class ScreenReader:
    """Mock ScreenReader class for testing"""
    def __init__(self, config):
        self.config = config
    
    def capture_game_state(self):
        """Return a mock game state"""
        return create_mock_game_state()
from src.decision.modified_agent import Agent
class Controller:
    """Mock Controller class for testing"""
    def __init__(self, config):
        self.config = config
    
    def execute(self, actions):
        """Mock execute method"""
        return True
def load_config():
    """Mock load_config function for testing"""
    return {
        "game_path": "C:\\Program Files (x86)\\World of Warcraft\\_retail_\\Wow.exe",
        "learning": {
            "enabled": True,
            "experience_buffer_size": 10000,
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "exploration_rate": 0.1,
            "save_interval": 300,
            "knowledge_queue_size": 1000,
            "min_confidence": 0.3,
            "inference_confidence": 0.6,
            "use_rl_for_decisions": False
        },
        "planning": {
            "enabled": True,
            "max_active_goals": 5,
            "goal_replan_interval": 300,
            "use_hierarchical_planning": True,
            "save_successful_plans": True,
            "plan_lookahead": 10
        }
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_learning_planning")

def create_mock_game_state():
    """Create a mock GameState for testing"""
    state = GameState()
    
    # Basic player attributes
    state.player_class = "warrior"
    state.level = 10
    state.health_percent = 85
    state.mana_percent = 65
    state.is_in_combat = False
    state.is_alive = True
    state.xp = 5000
    state.coordinates = (100, 200)
    state.current_zone = "elwynn_forest"
    state.is_in_group = False
    state.is_moving = False
    state.is_resting = False
    
    # Quest-related attributes
    state.active_quests = [
        {
            "id": "quest1",
            "name": "Wolves in the Forest",
            "description": "Kill 10 wolves in Elwynn Forest",
            "objectives": ["Kill 10 wolves (3/10)"],
            "status": "active"
        }
    ]
    state.current_quest = "quest1"
    state.quest_progress = 0.3  # 30% complete
    
    # Entities in the environment
    state.nearby_entities = [
        {
            "id": "wolf1",
            "name": "Forest Wolf",
            "type": "npc",
            "hostile": True,
            "level": 8,
            "position": (120, 220),
            "health_percent": 100
        },
        {
            "id": "npc1",
            "name": "Quest Giver",
            "type": "npc",
            "hostile": False,
            "interactable": True,
            "quest_giver": True,
            "position": (90, 180)
        },
        {
            "id": "herb1",
            "name": "Peacebloom",
            "type": "herb",
            "interactable": True,
            "position": (110, 190)
        }
    ]
    
    # Inventory items
    state.inventory = [
        {
            "name": "Hearthstone",
            "type": "consumable",
            "quality": "common"
        },
        {
            "name": "Wolf Meat",
            "type": "reagent",
            "quality": "common",
            "count": 5
        }
    ]
    
    # Usable items
    state.usable_items = ["Hearthstone"]
    
    return state

def test_reinforcement_learning(agent):
    """Test reinforcement learning functionality"""
    logger.info("Testing Reinforcement Learning...")
    
    # Create mock states
    state1 = create_mock_game_state()
    state2 = create_mock_game_state()
    state2.health_percent = 90  # Player gained health
    state2.quest_progress = 0.4  # Quest progressed
    
    # Make decisions in both states
    actions1 = agent.decide(state1)
    logger.info(f"Actions decided in state 1: {actions1}")
    
    # Record experience with a positive reward
    agent._record_experience(state2)
    
    # Learn from batch of experiences
    agent.learn_from_batch(batch_size=1)
    
    # Check if learning was stored
    logger.info(f"RL success: {len(agent.rl_manager.experience_buffer.buffer) > 0}")
    
    return len(agent.rl_manager.experience_buffer.buffer) > 0

def test_knowledge_expansion(agent):
    """Test knowledge expansion functionality"""
    logger.info("Testing Knowledge Expansion...")
    
    # Create mock state
    state = create_mock_game_state()
    
    # Process observations
    agent._process_observations(state)
    
    # Check if observations were processed
    expanded_npcs = agent.knowledge_manager.get_high_confidence_knowledge("npcs")
    expanded_locations = agent.knowledge_manager.get_high_confidence_knowledge("locations")
    
    logger.info(f"Knowledge expansion results:")
    logger.info(f"  NPCs: {list(expanded_npcs.keys())}")
    logger.info(f"  Locations: {list(expanded_locations.keys())}")
    
    return len(expanded_npcs) > 0 or len(expanded_locations) > 0

def test_transfer_learning(agent):
    """Test transfer learning functionality"""
    logger.info("Testing Transfer Learning...")
    
    # Register a transferable skill
    agent.register_transferable_skill(
        "shield_block_timing",
        "Optimal timing for using Shield Block ability",
        "warrior_combat",
        {"timing": "preemptive", "triggers": ["boss_swing", "high_damage_ability"]}
    )
    
    # Check if skill was registered
    skill = agent.transfer_manager.get_skill_by_name("shield_block_timing")
    
    # Test skill transfer to a similar context
    if skill:
        result = agent.transfer_manager.apply_skill("shield_block_timing", "paladin_combat")
        logger.info(f"Transfer result: {result}")
        
    return skill is not None

def test_hierarchical_planning(agent):
    """Test hierarchical planning functionality"""
    logger.info("Testing Hierarchical Planning...")
    
    # Create a quest plan
    quest_details = {
        "id": "quest1",
        "name": "Wolves in the Forest",
        "level_requirement": 5,
        "quest_giver": "Marshal Dughan",
        "objectives": [
            {"type": "kill", "target": "Forest Wolf", "count": 10}
        ]
    }
    
    plan = agent.generate_quest_plan(quest_details)
    logger.info(f"Generated quest plan with {len(plan)} steps")
    
    # Check if plan was generated
    return len(plan) > 0

def test_performance_metrics(agent):
    """Test performance metrics functionality"""
    logger.info("Testing Performance Metrics...")
    
    # Record some metrics
    agent.metrics_manager.record_metric("combat_dps", 150.5)
    agent.metrics_manager.record_metric("leveling_speed", 45.0)
    agent.metrics_manager.record_metric("quest_completion_time", 12.5)
    
    # Generate a report
    report = agent.generate_performance_report()
    logger.info(f"Performance report generated: {len(report)} characters")
    
    # Check metrics were recorded
    dps_metric = agent.metrics_manager.get_metric("combat_dps")
    
    return dps_metric is not None and dps_metric.get_latest_value() == 150.5

def main():
    """Main test function"""
    logger.info("Starting learning and planning system tests")
    
    # Load configuration
    config = load_config()
    
    # Create mock components
    screen_reader = ScreenReader(config)
    controller = Controller(config)
    
    # Create the agent
    agent = Agent(config, screen_reader, controller)
    
    # Run tests
    results = {
        "reinforcement_learning": test_reinforcement_learning(agent),
        "knowledge_expansion": test_knowledge_expansion(agent),
        "transfer_learning": test_transfer_learning(agent),
        "hierarchical_planning": test_hierarchical_planning(agent),
        "performance_metrics": test_performance_metrics(agent)
    }
    
    # Save the learning data
    agent._save_learning_data()
    
    # Output results
    logger.info("==== Test Results ====")
    for test, passed in results.items():
        logger.info(f"{test}: {'PASSED' if passed else 'FAILED'}")
    
    all_passed = all(results.values())
    logger.info(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()