"""
Central manager for economic intelligence, coordinating all economic subsystems.
"""
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from ..utils.config import Config
from .market_analyzer import MarketAnalyzer
from .farming_optimizer import FarmingOptimizer, FarmingRoute, ResourceNode
from .crafting_manager import CraftingManager, Recipe
from .inventory_manager import InventoryManager, InventoryItem, StorageLocation, ItemCategory

logger = logging.getLogger(__name__)

class EconomicManager:
    """
    Central coordinator for economic intelligence systems.
    
    This class integrates:
    - Market Analysis
    - Farming Optimization
    - Crafting Intelligence
    - Inventory Management
    
    It provides high-level economic decision making.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the EconomicManager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Initialize subsystems
        self.market_analyzer = MarketAnalyzer(config)
        self.farming_optimizer = FarmingOptimizer(config, self.market_analyzer)
        self.crafting_manager = CraftingManager(config, self.market_analyzer)
        self.inventory_manager = InventoryManager(config, self.market_analyzer)
        
        # Character info
        self.character_level = config.get("character.level", 60)
        self.profession_skills = config.get("character.professions", {})
        self.gold = config.get("character.gold", 0)
        
        # Last update timestamps
        self.last_market_update = None
        self.last_inventory_update = None
        
        logger.info("EconomicManager initialized")
        
    def update_market_data(self, auction_data: List[Dict]) -> None:
        """
        Update market data with new auction house information.
        
        Args:
            auction_data: List of auction items with pricing info
        """
        self.market_analyzer.update_prices(auction_data)
        self.last_market_update = datetime.now()
        logger.info(f"Updated market data with {len(auction_data)} items")
        
    def update_character_info(self, level: int, gold: int, profession_skills: Dict[str, int]) -> None:
        """
        Update character information.
        
        Args:
            level: Character level
            gold: Current gold
            profession_skills: Dictionary mapping profession names to skill levels
        """
        self.character_level = level
        self.gold = gold
        self.profession_skills = profession_skills
        logger.info(f"Updated character info: Level {level}, Gold {gold}, Professions {profession_skills}")
        
    def scan_inventory(self, inventory_data: List[Dict]) -> None:
        """
        Scan and update character inventory.
        
        Args:
            inventory_data: List of inventory items
        """
        # First, clear existing bag items
        for item_id in list(self.inventory_manager.items_by_location[StorageLocation.BAG]):
            self.inventory_manager.remove_item(item_id, self.inventory_manager.items[item_id].quantity)
            
        # Add new items
        for item_data in inventory_data:
            # Convert to InventoryItem
            item = InventoryItem(
                item_id=item_data["item_id"],
                name=item_data["name"],
                category=ItemCategory(item_data["category"]),
                stack_size=item_data["stack_size"],
                quantity=item_data["quantity"],
                unit_value=item_data.get("unit_value", 0),
                is_soulbound=item_data.get("is_soulbound", False),
                item_level=item_data.get("item_level", 1),
                required_level=item_data.get("required_level", 1),
                is_unique=item_data.get("is_unique", False),
                is_quest_item=item_data.get("is_quest_item", False)
            )
            
            self.inventory_manager.add_item(item)
            
        self.last_inventory_update = datetime.now()
        logger.info(f"Updated inventory with {len(inventory_data)} items")
        
    def add_resource_node(self, node_data: Dict) -> None:
        """
        Add a discovered resource node to the farming database.
        
        Args:
            node_data: Resource node data
        """
        node = ResourceNode(
            node_id=node_data["node_id"],
            resource_type=node_data["resource_type"],
            name=node_data["name"],
            x=node_data["x"],
            y=node_data["y"],
            zone=node_data["zone"],
            level_req=node_data.get("level_req", 1),
            skill_req=node_data.get("skill_req", 1)
        )
        
        self.farming_optimizer.add_node(node)
        logger.info(f"Added resource node: {node.name} in {node.zone}")
        
    def record_node_harvested(self, node_id: int) -> None:
        """
        Record that a resource node has been harvested.
        
        Args:
            node_id: ID of the harvested node
        """
        self.farming_optimizer.record_node_visit(node_id)
        logger.info(f"Recorded harvest of node {node_id}")
        
    def add_recipe(self, recipe_data: Dict) -> None:
        """
        Add a recipe to the crafting database.
        
        Args:
            recipe_data: Recipe data
        """
        # Create materials
        from .crafting_manager import Material
        
        materials = []
        for mat_data in recipe_data.get("materials", []):
            material = Material(
                item_id=mat_data["item_id"],
                name=mat_data["name"],
                quantity=mat_data["quantity"]
            )
            materials.append(material)
            
        # Create recipe
        recipe = Recipe(
            recipe_id=recipe_data["recipe_id"],
            name=recipe_data["name"],
            profession=recipe_data["profession"],
            skill_req=recipe_data["skill_req"],
            result_item_id=recipe_data["result_item_id"],
            result_item_name=recipe_data["result_item_name"],
            result_quantity=recipe_data.get("result_quantity", 1),
            materials=materials,
            crafting_cost=recipe_data.get("crafting_cost", 0),
            cooldown=recipe_data.get("cooldown", 0)
        )
        
        self.crafting_manager.add_recipe(recipe)
        logger.info(f"Added recipe: {recipe.name} for {recipe.profession}")
        
    def get_optimal_farming_routes(self, top_n: int = 3) -> List[Dict]:
        """
        Get the most profitable farming routes.
        
        Args:
            top_n: Number of routes to return
            
        Returns:
            List of farming route summaries
        """
        routes = self.farming_optimizer.get_best_farming_routes(
            character_level=self.character_level,
            profession_skills=self.profession_skills,
            top_n=top_n
        )
        
        return [route.summary for route in routes]
        
    def get_profitable_crafts(self, profession: Optional[str] = None, top_n: int = 10) -> List[Dict]:
        """
        Get the most profitable crafting options.
        
        Args:
            profession: Profession to filter by (None for all)
            top_n: Number of crafts to return
            
        Returns:
            List of profitable craft summaries
        """
        # Get skill level for the profession
        skill_level = 0
        if profession and profession in self.profession_skills:
            skill_level = self.profession_skills[profession]
            
        # Get profitable recipes
        recipes = self.crafting_manager.get_profitable_recipes(
            profession=profession,
            min_skill=0,
            min_profit=0,
            max_recipes=top_n
        )
        
        # Filter by skill level
        available_recipes = [r for r in recipes if r.skill_req <= skill_level]
        
        # Create summaries
        result = []
        for recipe in available_recipes:
            result.append({
                "name": recipe.name,
                "profession": recipe.profession,
                "skill_required": recipe.skill_req,
                "materials_cost": recipe.total_cost,
                "market_value": recipe.market_value,
                "profit": recipe.profit_per_craft,
                "profit_margin": f"{recipe.profit_margin:.1f}%"
            })
            
        return result
        
    def get_crafting_shopping_list(self, recipe_ids: List[int], quantities: List[int]) -> Dict:
        """
        Generate a shopping list for crafting multiple items.
        
        Args:
            recipe_ids: List of recipe IDs
            quantities: List of quantities to craft
            
        Returns:
            Dictionary with shopping list details
        """
        if len(recipe_ids) != len(quantities):
            logger.error("Recipe IDs and quantities lists must be the same length")
            return {"error": "Invalid input"}
            
        # Create combined shopping list
        shopping_list = {}
        total_cost = 0
        recipes_info = []
        
        for i, recipe_id in enumerate(recipe_ids):
            quantity = quantities[i]
            
            if recipe_id not in self.crafting_manager.recipes:
                logger.warning(f"Recipe {recipe_id} not found")
                continue
                
            recipe = self.crafting_manager.recipes[recipe_id]
            
            # Update recipe prices
            self.crafting_manager._update_recipe_prices(recipe)
            
            # Add to recipe info
            recipes_info.append({
                "name": recipe.name,
                "quantity": quantity,
                "cost_per_craft": recipe.total_cost,
                "total_cost": recipe.total_cost * quantity
            })
            
            # Add materials to shopping list
            for material in recipe.materials:
                mat_quantity = material.quantity * quantity
                if material.name in shopping_list:
                    shopping_list[material.name]["quantity"] += mat_quantity
                else:
                    shopping_list[material.name] = {
                        "item_id": material.item_id,
                        "name": material.name,
                        "quantity": mat_quantity,
                        "unit_price": material.unit_price,
                        "total_price": material.unit_price * mat_quantity
                    }
                    
            # Add to total cost
            total_cost += recipe.total_cost * quantity
            
        # Convert shopping list to list
        materials_list = list(shopping_list.values())
        materials_list.sort(key=lambda x: x["total_price"], reverse=True)
        
        return {
            "recipes": recipes_info,
            "materials": materials_list,
            "total_cost": total_cost,
            "affordable": total_cost <= self.gold
        }
        
    def optimize_inventory(self) -> Dict:
        """
        Optimize inventory organization and make sell/keep recommendations.
        
        Returns:
            Dictionary with inventory optimization recommendations
        """
        # Get optimization recommendations
        recommendations = self.inventory_manager.optimize_inventory()
        
        # Create summary
        result = {
            "vendor": [],
            "auction": [],
            "bank": [],
            "discard": [],
            "estimated_vendor_value": 0,
            "estimated_auction_value": 0
        }
        
        # Process recommendations
        for location, items in recommendations.items():
            if location == StorageLocation.VENDOR:
                result["vendor"] = [{
                    "name": item.name,
                    "quantity": item.quantity,
                    "value": item.total_value
                } for item in items]
                result["estimated_vendor_value"] = sum(item.total_value for item in items)
                
            elif location == StorageLocation.AUCTION:
                result["auction"] = [{
                    "name": item.name,
                    "quantity": item.quantity,
                    "value": item.total_value
                } for item in items]
                result["estimated_auction_value"] = sum(item.total_value for item in items)
                
            elif location == StorageLocation.BANK:
                result["bank"] = [{
                    "name": item.name,
                    "quantity": item.quantity,
                    "value": item.total_value
                } for item in items]
                
            elif location == StorageLocation.DISCARD:
                result["discard"] = [{
                    "name": item.name,
                    "quantity": item.quantity
                } for item in items]
                
        return result
        
    def make_gold_earning_plan(self) -> Dict:
        """
        Create a comprehensive plan for earning gold.
        
        Returns:
            Dictionary with gold earning recommendations
        """
        plan = {
            "current_gold": self.gold,
            "farming": {},
            "crafting": {},
            "inventory": {},
            "total_potential": 0
        }
        
        # 1. Get best farming routes
        farming_routes = self.farming_optimizer.get_best_farming_routes(
            character_level=self.character_level,
            profession_skills=self.profession_skills,
            top_n=3
        )
        
        if farming_routes:
            plan["farming"] = {
                "routes": [route.summary for route in farming_routes],
                "best_gold_per_hour": int(max(route.gold_per_hour for route in farming_routes))
            }
            
        # 2. Get profitable crafts
        profitable_crafts = []
        
        for profession, skill in self.profession_skills.items():
            if skill > 0:
                recipes = self.crafting_manager.get_profitable_recipes(
                    profession=profession,
                    min_skill=0,
                    min_profit_margin=10.0,  # At least 10% profit margin
                    max_recipes=5
                )
                
                for recipe in recipes:
                    if recipe.skill_req <= skill:
                        profitable_crafts.append({
                            "name": recipe.name,
                            "profession": recipe.profession,
                            "profit_per_craft": recipe.profit_per_craft,
                            "profit_margin": recipe.profit_margin,
                            "materials_cost": recipe.total_cost
                        })
                        
        if profitable_crafts:
            # Sort by profit per craft
            profitable_crafts.sort(key=lambda x: x["profit_per_craft"], reverse=True)
            plan["crafting"] = {
                "profitable_crafts": profitable_crafts[:5],  # Top 5
                "total_crafts": len(profitable_crafts)
            }
            
        # 3. Inventory optimization
        if self.last_inventory_update:
            # Get vendor vs AH analysis
            inventory_analysis = self.inventory_manager.vendor_vs_ah_analysis()
            
            plan["inventory"] = {
                "vendor_value": inventory_analysis["vendor_total"],
                "auction_value": inventory_analysis["auction_net"],
                "vendor_items_count": len(inventory_analysis["vendor_items"]),
                "auction_items_count": len(inventory_analysis["auction_items"])
            }
            
        # Calculate total potential gold
        plan["total_potential"] = (
            plan["inventory"].get("vendor_value", 0) +
            plan["inventory"].get("auction_value", 0)
        )
        
        # Add recommendations based on professions and inventory
        recommendations = []
        
        # If we have good farming professions
        gathering_professions = {"herbalism", "mining", "skinning"}
        if any(prof in gathering_professions for prof in self.profession_skills):
            if farming_routes:
                recommendations.append({
                    "priority": "high" if plan["farming"].get("best_gold_per_hour", 0) > 5000 else "medium",
                    "action": "farming",
                    "details": f"Farm in {farming_routes[0].zone} for approximately {int(farming_routes[0].gold_per_hour)} gold per hour"
                })
                
        # If we have crafting professions with profitable recipes
        if profitable_crafts:
            best_craft = profitable_crafts[0]
            recommendations.append({
                "priority": "high" if best_craft["profit_margin"] > 50 else "medium",
                "action": "crafting",
                "details": f"Craft {best_craft['name']} for {int(best_craft['profit_per_craft']/10000)} gold profit per craft ({best_craft['profit_margin']:.1f}% margin)"
            })
            
        # If we have items to sell
        if plan["inventory"].get("auction_value", 0) > 0:
            recommendations.append({
                "priority": "medium",
                "action": "sell_auction",
                "details": f"Sell {plan['inventory'].get('auction_items_count', 0)} items on the auction house for approximately {int(plan['inventory'].get('auction_value', 0)/10000)} gold"
            })
            
        if plan["inventory"].get("vendor_value", 0) > 0:
            recommendations.append({
                "priority": "low",
                "action": "sell_vendor",
                "details": f"Sell {plan['inventory'].get('vendor_items_count', 0)} items to vendors for approximately {int(plan['inventory'].get('vendor_value', 0)/10000)} gold"
            })
            
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])
        
        plan["recommendations"] = recommendations
        
        return plan