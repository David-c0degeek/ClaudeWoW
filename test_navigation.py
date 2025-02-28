import logging

def setup():
    """
    Test the navigation_manager fix
    """
    from src.knowledge.game_knowledge import GameKnowledge
    from src.perception.screen_reader import GameState
    from src.decision.navigation_manager import NavigationManager
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create game knowledge
    config = {}
    knowledge = GameKnowledge(config)
    
    # Create navigation manager
    nav_manager = NavigationManager(config, knowledge)
    
    # Create game state with player position
    state = GameState()
    state.player_position = (100, 100)
    state.current_zone = "test_zone"
    
    # Test navigation
    destination = (200, 200)
    
    try:
        path = nav_manager.generate_navigation_plan(state, destination)
        logging.info(f"Generated path: {path}")
        return True
    except Exception as e:
        logging.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    setup()

