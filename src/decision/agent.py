"""
Agent Module with Economic Intelligence Support
"""

import logging
from typing import Dict, List, Any

from src.perception.screen_reader import GameState, ScreenReader
from src.action.controller import Controller
from src.utils.config import Config
from src.decision.navigation_integration import NavigationSystem
from src.economic.economic_manager import EconomicManager

class Agent:
    """
    Main decision-making agent with advanced navigation and economic intelligence
    """
    
    def __init__(self, config: Dict, screen_reader: ScreenReader, controller: Controller, 
                 navigation_system=None, economic_manager=None):
        """
        Initialize the Agent
        
        Args:
            config: Configuration dictionary
            screen_reader: ScreenReader instance for perceiving the game
            controller: Controller instance for executing actions
            navigation_system: Optional NavigationSystem instance
            economic_manager: Optional EconomicManager instance
        """
        self.logger = logging.getLogger("wow_ai.decision.agent")
        self.config = config
        self.screen_reader = screen_reader
        self.controller = controller
        
        # Use provided navigation system
        self.navigation_system = navigation_system
        
        # Use provided economic manager or create one if not provided
        if economic_manager:
            self.economic_manager = economic_manager
        else:
            self.logger.info("No economic manager provided, initializing a new one")
            self.economic_manager = EconomicManager(config)
            
        self.logger.info("Agent initialized with navigation and economic intelligence")
    
    def decide(self, game_state: GameState) -> List[Dict[str, Any]]:
        """
        Make decisions based on the current game state
        
        Args:
            game_state: Current game state from perception system
        
        Returns:
            List[Dict]: List of actions to execute
        """
        self.logger.debug("Making decision based on current game state")
        
        # Process economic information if available
        if "inventory" in game_state:
            self.economic_manager.scan_inventory(game_state["inventory"])
            
        if "at_auction_house" in game_state and game_state["at_auction_house"]:
            if "auction_data" in game_state:
                self.economic_manager.update_market_data(game_state["auction_data"])
        
        # Basic placeholder decision - in a real implementation this would use
        # the navigation system and economic manager to make decisions
        return [{"type": "wait", "duration": 0.1}]
    
    def get_economic_insights(self) -> Dict[str, Any]:
        """
        Get economic insights from the economic manager
        
        Returns:
            Dict[str, Any]: Economic insights
        """
        # Generate a gold earning plan
        gold_plan = self.economic_manager.make_gold_earning_plan()
        
        # Get optimal farming routes
        farming_routes = self.economic_manager.get_optimal_farming_routes()
        
        # Get inventory optimization recommendations
        inventory_recommendations = self.economic_manager.optimize_inventory()
        
        # Get profitable crafting options for all professions
        profitable_crafts = []
        for profession in self.economic_manager.profession_skills.keys():
            if self.economic_manager.profession_skills[profession] > 0:
                profession_crafts = self.economic_manager.get_profitable_crafts(profession=profession)
                if profession_crafts:
                    profitable_crafts.extend(profession_crafts)
        
        # Combine all insights
        insights = {
            "gold_plan": gold_plan,
            "farming_routes": farming_routes[:3],  # Top 3 routes
            "inventory_recommendations": inventory_recommendations,
            "profitable_crafts": profitable_crafts[:5]  # Top 5 crafts
        }
        
        return insights