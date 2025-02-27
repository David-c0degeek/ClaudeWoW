"""
WoW AI Player - Main Entry Point
"""
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.screen_reader import ScreenReader
from src.decision.agent import Agent
from src.action.controller import Controller
from src.utils.config import load_config

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
    logger.info("Starting WoW AI Player")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    try:
        # Initialize components
        screen_reader = ScreenReader(config)
        controller = Controller(config)
        agent = Agent(config, screen_reader, controller)
        
        logger.info("All components initialized successfully")
        
        # Main loop
        logger.info("Entering main loop")
        while True:
            # Process game state
            game_state = screen_reader.capture_game_state()
            
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
        logger.info("WoW AI Player shutting down")

if __name__ == "__main__":
    main()
