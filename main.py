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
from src.economic.economic_manager import EconomicManager

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
        navigation_system = NavigationSystem(config, screen_reader)
        logger.info("Advanced navigation system initialized")
        
        # Initialize economic intelligence system
        economic_manager = EconomicManager(config)
        logger.info("Economic intelligence system initialized")
        
        # Initialize agent with navigation system and economic intelligence
        agent = Agent(config, screen_reader, controller, navigation_system, economic_manager=economic_manager)
        
        logger.info("All components initialized successfully")
        
        # Create required data directories if they don't exist
        data_dirs = [
            "data/economic",
            "data/economic/price_data",
            "data/economic/resource_nodes",
            "data/economic/recipes"
        ]
        for directory in data_dirs:
            os.makedirs(directory, exist_ok=True)
        
        # Main loop
        logger.info("Entering main loop")
        last_economic_update = time.time()
        economic_update_interval = config.get("economic.update_interval", 3600)  # 1 hour default
        
        while True:
            # Process game state
            game_state = screen_reader.capture_game_state()
            
            # Periodic economic data updates
            current_time = time.time()
            if current_time - last_economic_update > economic_update_interval:
                logger.info("Performing periodic economic data update")
                
                # Update character info (gold, profession skills, etc.)
                if "character_info" in game_state:
                    economic_manager.update_character_info(
                        level=game_state["character_info"].get("level", 60),
                        gold=game_state["character_info"].get("gold", 0),
                        profession_skills=game_state["character_info"].get("profession_skills", {})
                    )
                
                # Scan inventory if available
                if "inventory" in game_state:
                    economic_manager.scan_inventory(game_state["inventory"])
                
                # Update auction house data if we're at the AH
                if "at_auction_house" in game_state and game_state["at_auction_house"]:
                    if "auction_data" in game_state:
                        economic_manager.update_market_data(game_state["auction_data"])
                
                last_economic_update = current_time
            
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