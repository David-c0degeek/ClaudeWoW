"""
ClaudeWoW - Advanced AI Player for World of Warcraft
Main Entry Point
"""
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.screen_reader import ScreenReader
# Use our enhanced agent with advanced navigation capabilities
from src.decision.modified_agent import Agent
from src.action.controller import Controller
from src.utils.config import load_config
from src.decision.navigation_integration import NavigationSystem

def setup_logging():
    """Configure logging"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"wow_ai_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("wow_ai")

def is_game_client_running(game_state):
    """
    Check if a World of Warcraft client is actually running
    
    Args:
        game_state: Current game state
        
    Returns:
        bool: True if game client appears to be running, False otherwise
    """
    # Check for key indicators that a game client is running:
    # 1. Has valid player position
    if hasattr(game_state, "player_position") and any(game_state.player_position):
        return True
    
    # 2. Has non-empty zone information
    if hasattr(game_state, "current_zone") and game_state.current_zone:
        return True
    
    # 3. Has some UI elements detected
    if hasattr(game_state, "minimap_data") and game_state.minimap_data:
        return True
    
    # 4. Has valid player health/mana
    if hasattr(game_state, "player_health") and game_state.player_health != 100.0:
        return True
        
    # No indicators found - game client probably not running
    return False

def main():
    """Main entry point for the WoW AI Player"""
    logger = setup_logging()
    logger.info("Starting ClaudeWoW AI Player")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    try:
        # Initialize components
        screen_reader = ScreenReader(config)
        controller = Controller(config)
        
        # Initialize navigation system
        from src.knowledge.game_knowledge import GameKnowledge
        knowledge = GameKnowledge(config)
        navigation_system = NavigationSystem(config, knowledge)
        logger.info("Advanced navigation system initialized")
        
        # Initialize agent
        agent = Agent(config, screen_reader, controller)
        
        # Set navigation system
        agent.navigation_manager = navigation_system
        
        logger.info("All components initialized successfully")
        
        # Create required data directories if they don't exist
        data_dirs = [
            "data/game_knowledge",
            "data/recordings",
            "data/models/learning/plans"
        ]
        for directory in data_dirs:
            os.makedirs(directory, exist_ok=True)
        
        # Main loop
        logger.info("Entering main loop")
        
        # Track if we've warned about no game client
        client_warning_shown = False
        
        while True:
            # Process game state
            game_state = screen_reader.capture_game_state()
            
            # Check if a game client is actually running
            if not is_game_client_running(game_state):
                if not client_warning_shown:
                    logger.warning("No World of Warcraft client detected - AI in standby mode")
                    client_warning_shown = True
                time.sleep(2.0)  # Sleep longer when no client is detected
                continue
            else:
                if client_warning_shown:
                    logger.info("World of Warcraft client detected - AI resuming normal operation")
                    client_warning_shown = False
            
            # Decide on actions
            actions = agent.decide(game_state)
            
            # Execute actions
            controller.execute(actions)
            
            # Sleep to prevent high CPU usage
            time.sleep(config.get("loop_interval", 0.1))
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
    finally:
        logger.info("ClaudeWoW AI Player shutting down")

if __name__ == "__main__":
    main()