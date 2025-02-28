"""
ClaudeWoW Economic Intelligence System Demo

This script demonstrates the Economic Intelligence System capabilities by:
1. Loading and initializing the Economic Manager
2. Generating test data to simulate game data
3. Showcasing key economic decision-making features
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List

from src.utils.config import load_config
from src.economic.economic_manager import EconomicManager
from src.economic.market_analyzer import MarketAnalyzer
from src.economic.farming_optimizer import FarmingOptimizer, ResourceNode
from src.economic.crafting_manager import CraftingManager, Recipe, Material
from src.economic.inventory_manager import InventoryManager, InventoryItem, ItemCategory, StorageLocation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("economic_demo")

def generate_test_auction_data() -> List[Dict]:
    """Generate test auction house data."""
    auction_data = [
        {
            "id": 1,
            "name": "Dreamfoil",
            "buyout_price": 5000,  # 50 silver
            "quantity": 20
        },
        {
            "id": 2,
            "name": "Mountain Silversage",
            "buyout_price": 9000,  # 90 silver
            "quantity": 15
        },
        {
            "id": 3,
            "name": "Black Lotus",
            "buyout_price": 1000000,  # 100 gold
            "quantity": 1
        },
        {
            "id": 4,
            "name": "Thorium Ore",
            "buyout_price": 3500,  # 35 silver
            "quantity": 20
        },
        {
            "id": 5,
            "name": "Arcanite Bar",
            "buyout_price": 450000,  # 45 gold
            "quantity": 1
        },
        {
            "id": 6,
            "name": "Flask of Supreme Power",
            "buyout_price": 1500000,  # 150 gold
            "quantity": 1
        },
        {
            "id": 7,
            "name": "Major Healing Potion",
            "buyout_price": 30000,  # 3 gold
            "quantity": 5
        },
        {
            "id": 8,
            "name": "Crystal Vial",
            "buyout_price": 5000,  # 50 silver
            "quantity": 10
        },
        {
            "id": 10,
            "name": "Arcane Crystal",
            "buyout_price": 350000,  # 35 gold
            "quantity": 1
        },
        {
            "id": 11,
            "name": "Thorium Bar",
            "buyout_price": 20000,  # 2 gold
            "quantity": 5
        }
    ]
    return auction_data

def generate_test_resource_nodes() -> List[Dict]:
    """Generate test resource nodes data."""
    nodes = [
        {
            "node_id": 1,
            "resource_type": "herb",
            "name": "Dreamfoil",
            "x": 1500.5,
            "y": 2400.8,
            "zone": "Un'Goro Crater",
            "level_req": 55,
            "skill_req": 250
        },
        {
            "node_id": 2,
            "resource_type": "herb",
            "name": "Mountain Silversage",
            "x": 1600.2,
            "y": 2500.3,
            "zone": "Un'Goro Crater",
            "level_req": 58,
            "skill_req": 280
        },
        {
            "node_id": 3,
            "resource_type": "herb",
            "name": "Black Lotus",
            "x": 1700.7,
            "y": 2600.9,
            "zone": "Eastern Plaguelands",
            "level_req": 60,
            "skill_req": 300
        },
        {
            "node_id": 4,
            "resource_type": "ore",
            "name": "Thorium Vein",
            "x": 1200.3,
            "y": 2100.5,
            "zone": "Un'Goro Crater",
            "level_req": 55,
            "skill_req": 250
        },
        {
            "node_id": 5,
            "resource_type": "ore",
            "name": "Rich Thorium Vein",
            "x": 1300.8,
            "y": 2200.1,
            "zone": "Eastern Plaguelands",
            "level_req": 58,
            "skill_req": 275
        }
    ]
    return nodes

def generate_test_recipe_data() -> List[Dict]:
    """Generate test recipe data."""
    recipes = [
        {
            "recipe_id": 1,
            "name": "Flask of Supreme Power",
            "profession": "alchemy",
            "skill_req": 300,
            "result_item_id": 6,
            "result_item_name": "Flask of Supreme Power",
            "result_quantity": 1,
            "crafting_cost": 10000,  # 1 gold vendor cost
            "materials": [
                {"item_id": 1, "name": "Dreamfoil", "quantity": 30},
                {"item_id": 2, "name": "Mountain Silversage", "quantity": 10},
                {"item_id": 3, "name": "Black Lotus", "quantity": 1},
                {"item_id": 8, "name": "Crystal Vial", "quantity": 1}
            ]
        },
        {
            "recipe_id": 2,
            "name": "Major Healing Potion",
            "profession": "alchemy",
            "skill_req": 275,
            "result_item_id": 7,
            "result_item_name": "Major Healing Potion",
            "result_quantity": 1,
            "crafting_cost": 1000,  # 10 silver vendor cost
            "materials": [
                {"item_id": 1, "name": "Dreamfoil", "quantity": 2},
                {"item_id": 2, "name": "Mountain Silversage", "quantity": 1},
                {"item_id": 8, "name": "Crystal Vial", "quantity": 1}
            ]
        },
        {
            "recipe_id": 3,
            "name": "Arcanite Bar",
            "profession": "alchemy",
            "skill_req": 275,
            "result_item_id": 5,
            "result_item_name": "Arcanite Bar",
            "result_quantity": 1,
            "crafting_cost": 0,
            "materials": [
                {"item_id": 10, "name": "Arcane Crystal", "quantity": 1},
                {"item_id": 11, "name": "Thorium Bar", "quantity": 1}
            ]
        }
    ]
    return recipes

def generate_test_inventory() -> List[Dict]:
    """Generate test inventory data."""
    items = [
        {
            "item_id": 1,
            "name": "Dreamfoil",
            "category": "material",
            "stack_size": 20,
            "quantity": 25,
            "unit_value": 5000,
            "is_soulbound": False
        },
        {
            "item_id": 2,
            "name": "Mountain Silversage",
            "category": "material",
            "stack_size": 20,
            "quantity": 15,
            "unit_value": 9000,
            "is_soulbound": False
        },
        {
            "item_id": 3,
            "name": "Black Lotus",
            "category": "material",
            "stack_size": 10,
            "quantity": 2,
            "unit_value": 1000000,
            "is_soulbound": False
        },
        {
            "item_id": 5,
            "name": "Arcanite Bar",
            "category": "material",
            "stack_size": 20,
            "quantity": 5,
            "unit_value": 450000,
            "is_soulbound": False
        },
        {
            "item_id": 12,
            "name": "Tier 2 Helmet",
            "category": "equipment",
            "stack_size": 1,
            "quantity": 1,
            "unit_value": 1000000,
            "is_soulbound": True,
            "item_level": 66,
            "required_level": 60
        },
        {
            "item_id": 13,
            "name": "Rubbish Item",
            "category": "junk",
            "stack_size": 10,
            "quantity": 5,
            "unit_value": 10,
            "is_soulbound": False
        }
    ]
    return items

def main():
    """Main entry point for the Economic Intelligence demo."""
    logger.info("Starting ClaudeWoW Economic Intelligence Demo")
    
    # Create data directories if they don't exist
    os.makedirs("data/economic", exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Update config specifically for demo
    config_data = config.data if hasattr(config, "data") else {}
    if not config_data:
        config_data = {}
        
    config_data["character"] = {
        "level": 60,
        "gold": 10000 * 10000,  # 10,000 gold in copper
        "professions": {
            "herbalism": 300,
            "alchemy": 300,
            "mining": 150
        }
    }
    
    config_data["paths"] = {
        "economic": {
            "price_data": "data/economic/demo_price_data.json",
            "nodes_data": "data/economic/demo_resource_nodes.json",
            "recipes_data": "data/economic/demo_recipes.json",
            "inventory_data": "data/economic/demo_inventory.json",
            "container_data": "data/economic/demo_containers.json"
        }
    }
    
    config_data["inventory"] = {
        "vendor_threshold": 100,
        "ah_threshold": 500,
        "keep_threshold": 5000
    }
    
    config.data = config_data
    
    # Initialize the economic manager
    economic_manager = EconomicManager(config)
    logger.info("Initialized Economic Manager")
    
    # Update character information
    economic_manager.update_character_info(
        level=60,
        gold=10000 * 10000,  # 10,000 gold
        profession_skills={"herbalism": 300, "alchemy": 300, "mining": 150}
    )
    
    # Add test auction data
    auction_data = generate_test_auction_data()
    economic_manager.update_market_data(auction_data)
    logger.info(f"Added {len(auction_data)} auction items")
    
    # Add test resource nodes
    for node_data in generate_test_resource_nodes():
        economic_manager.add_resource_node(node_data)
    logger.info("Added resource nodes")
    
    # Add test recipes
    for recipe_data in generate_test_recipe_data():
        economic_manager.add_recipe(recipe_data)
    logger.info("Added crafting recipes")
    
    # Add test inventory
    inventory_data = generate_test_inventory()
    economic_manager.scan_inventory(inventory_data)
    logger.info(f"Scanned {len(inventory_data)} inventory items")
    
    # Now demonstrate economic intelligence capabilities
    logger.info("\n----- ECONOMIC INTELLIGENCE DEMONSTRATION -----\n")
    
    # 1. Get optimal farming routes
    print("\n=== OPTIMAL FARMING ROUTES ===")
    farming_routes = economic_manager.get_optimal_farming_routes(top_n=2)
    for i, route in enumerate(farming_routes, 1):
        print(f"Route {i}: {route['zone']}")
        print(f"  Gold per hour: {route['gold_per_hour']/10000:.2f}g")
        print(f"  Nodes: {route['nodes']}")
        print(f"  Resource types: {', '.join(route['resource_types'])}")
        print(f"  Estimated time: {route['time_minutes']:.1f} minutes")
    
    # 2. Get profitable crafts
    print("\n=== PROFITABLE CRAFTING OPTIONS ===")
    alchemy_crafts = economic_manager.get_profitable_crafts(profession="alchemy", top_n=3)
    for i, craft in enumerate(alchemy_crafts, 1):
        print(f"Craft {i}: {craft['name']}")
        print(f"  Materials cost: {craft['materials_cost']/10000:.2f}g")
        print(f"  Market value: {craft['market_value']/10000:.2f}g")
        print(f"  Profit: {craft['profit']/10000:.2f}g ({craft['profit_margin']})")
    
    # 3. Get inventory optimization recommendations
    print("\n=== INVENTORY OPTIMIZATION ===")
    inventory_recommendations = economic_manager.optimize_inventory()
    
    if inventory_recommendations["vendor"]:
        print(f"Vendor: {len(inventory_recommendations['vendor'])} items")
        for item in inventory_recommendations["vendor"][:2]:  # Show top 2
            print(f"  - {item['name']} x{item['quantity']}: {item['value']/10000:.2f}g")
        
    if inventory_recommendations["auction"]:
        print(f"Auction: {len(inventory_recommendations['auction'])} items")
        for item in inventory_recommendations["auction"][:2]:  # Show top 2
            print(f"  - {item['name']} x{item['quantity']}: {item['value']/10000:.2f}g")
    
    # 4. Get crafting shopping list
    print("\n=== CRAFTING SHOPPING LIST ===")
    shopping_list = economic_manager.get_crafting_shopping_list([1], [5])  # 5x Flask of Supreme Power
    
    if "recipes" in shopping_list:
        print("Recipes to craft:")
        for recipe in shopping_list["recipes"]:
            print(f"  - {recipe['quantity']}x {recipe['name']}: {recipe['total_cost']/10000:.2f}g")
            
    if "materials" in shopping_list:
        print("Materials needed:")
        for material in shopping_list["materials"]:
            print(f"  - {material['quantity']}x {material['name']}: {material['total_price']/10000:.2f}g")
            
    print(f"Total cost: {shopping_list.get('total_cost', 0)/10000:.2f}g")
    print(f"Affordable: {shopping_list.get('affordable', False)}")
    
    # 5. Generate comprehensive gold earning plan
    print("\n=== GOLD EARNING PLAN ===")
    gold_plan = economic_manager.make_gold_earning_plan()
    
    print(f"Current gold: {gold_plan['current_gold']/10000:.2f}g")
    
    if "recommendations" in gold_plan:
        print("Recommendations:")
        for i, recommendation in enumerate(gold_plan["recommendations"], 1):
            print(f"  {i}. ({recommendation['priority']}) {recommendation['action']}: {recommendation['details']}")
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main()