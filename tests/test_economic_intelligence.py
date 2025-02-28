"""
Test file for Economic Intelligence System components.
"""
import sys
import os
import json
from datetime import datetime
from typing import Dict, List

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.economic.market_analyzer import MarketAnalyzer, ItemPrice
from src.economic.farming_optimizer import FarmingOptimizer, ResourceNode
from src.economic.crafting_manager import CraftingManager, Recipe, Material
from src.economic.inventory_manager import (
    InventoryManager, InventoryItem, StorageContainer, 
    ItemCategory, StorageLocation
)
from src.economic.economic_manager import EconomicManager

def get_test_config() -> Config:
    """Create a test configuration."""
    config_data = {
        "paths": {
            "economic": {
                "price_data": "data/economic/test_price_data.json",
                "nodes_data": "data/economic/test_resource_nodes.json",
                "recipes_data": "data/economic/test_recipes.json",
                "inventory_data": "data/economic/test_inventory.json",
                "container_data": "data/economic/test_containers.json"
            }
        },
        "character": {
            "level": 60,
            "gold": 10000 * 10000,  # 10,000 gold in copper
            "professions": {
                "herbalism": 300,
                "alchemy": 300,
                "mining": 150
            },
            "movement_speed": 7.0
        },
        "inventory": {
            "vendor_threshold": 100,
            "ah_threshold": 500,
            "keep_threshold": 5000
        }
    }
    
    return Config(config_data)

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
        }
    ]
    
    return auction_data

def generate_test_resource_nodes() -> List[ResourceNode]:
    """Generate test resource nodes."""
    nodes = [
        ResourceNode(
            node_id=1,
            resource_type="herb",
            name="Dreamfoil",
            x=1500.5,
            y=2400.8,
            zone="Un'Goro Crater",
            level_req=55,
            skill_req=250
        ),
        ResourceNode(
            node_id=2,
            resource_type="herb",
            name="Mountain Silversage",
            x=1600.2,
            y=2500.3,
            zone="Un'Goro Crater",
            level_req=58,
            skill_req=280
        ),
        ResourceNode(
            node_id=3,
            resource_type="herb",
            name="Black Lotus",
            x=1700.7,
            y=2600.9,
            zone="Eastern Plaguelands",
            level_req=60,
            skill_req=300
        ),
        ResourceNode(
            node_id=4,
            resource_type="ore",
            name="Thorium Vein",
            x=1200.3,
            y=2100.5,
            zone="Un'Goro Crater",
            level_req=55,
            skill_req=250
        ),
        ResourceNode(
            node_id=5,
            resource_type="ore",
            name="Rich Thorium Vein",
            x=1300.8,
            y=2200.1,
            zone="Eastern Plaguelands",
            level_req=58,
            skill_req=275
        )
    ]
    
    return nodes

def generate_test_recipes() -> List[Recipe]:
    """Generate test recipes."""
    recipes = [
        Recipe(
            recipe_id=1,
            name="Flask of Supreme Power",
            profession="alchemy",
            skill_req=300,
            result_item_id=6,
            result_item_name="Flask of Supreme Power",
            result_quantity=1,
            materials=[
                Material(item_id=1, name="Dreamfoil", quantity=30),
                Material(item_id=2, name="Mountain Silversage", quantity=10),
                Material(item_id=3, name="Black Lotus", quantity=1),
                Material(item_id=8, name="Crystal Vial", quantity=1, unit_price=5000)
            ],
            crafting_cost=10000  # 1 gold vendor cost
        ),
        Recipe(
            recipe_id=2,
            name="Major Healing Potion",
            profession="alchemy",
            skill_req=275,
            result_item_id=7,
            result_item_name="Major Healing Potion",
            result_quantity=1,
            materials=[
                Material(item_id=1, name="Dreamfoil", quantity=2),
                Material(item_id=2, name="Mountain Silversage", quantity=1),
                Material(item_id=9, name="Crystal Vial", quantity=1, unit_price=5000)
            ],
            crafting_cost=1000  # 10 silver vendor cost
        ),
        Recipe(
            recipe_id=3,
            name="Arcanite Bar",
            profession="alchemy",
            skill_req=275,
            result_item_id=5,
            result_item_name="Arcanite Bar",
            result_quantity=1,
            materials=[
                Material(item_id=10, name="Arcane Crystal", quantity=1, unit_price=350000),
                Material(item_id=11, name="Thorium Bar", quantity=1, unit_price=20000)
            ],
            crafting_cost=0
        )
    ]
    
    return recipes

def generate_test_inventory() -> List[InventoryItem]:
    """Generate test inventory items."""
    items = [
        InventoryItem(
            item_id=1,
            name="Dreamfoil",
            category=ItemCategory.MATERIAL,
            stack_size=20,
            quantity=25,
            unit_value=5000,
            is_soulbound=False,
            location=StorageLocation.BAG
        ),
        InventoryItem(
            item_id=2,
            name="Mountain Silversage",
            category=ItemCategory.MATERIAL,
            stack_size=20,
            quantity=15,
            unit_value=9000,
            is_soulbound=False,
            location=StorageLocation.BAG
        ),
        InventoryItem(
            item_id=3,
            name="Black Lotus",
            category=ItemCategory.MATERIAL,
            stack_size=10,
            quantity=2,
            unit_value=1000000,
            is_soulbound=False,
            location=StorageLocation.BAG
        ),
        InventoryItem(
            item_id=5,
            name="Arcanite Bar",
            category=ItemCategory.MATERIAL,
            stack_size=20,
            quantity=5,
            unit_value=450000,
            is_soulbound=False,
            location=StorageLocation.BAG
        ),
        InventoryItem(
            item_id=12,
            name="Tier 2 Helmet",
            category=ItemCategory.EQUIPMENT,
            stack_size=1,
            quantity=1,
            unit_value=1000000,
            is_soulbound=True,
            item_level=66,
            required_level=60,
            location=StorageLocation.BAG
        ),
        InventoryItem(
            item_id=13,
            name="Rubbish Item",
            category=ItemCategory.JUNK,
            stack_size=10,
            quantity=5,
            unit_value=10,
            is_soulbound=False,
            location=StorageLocation.BAG
        )
    ]
    
    return items

def test_market_analyzer():
    """Test market analyzer functionality."""
    print("\n=== Testing Market Analyzer ===")
    
    config = get_test_config()
    market_analyzer = MarketAnalyzer(config)
    
    # Update with test auction data
    auction_data = generate_test_auction_data()
    market_analyzer.update_prices(auction_data)
    
    # Test price trend analysis
    dreamfoil_trend = market_analyzer.get_item_trend(1)
    print(f"Dreamfoil trend: {dreamfoil_trend['trend']}")
    
    # Test arbitrage opportunities
    opportunities = market_analyzer.find_arbitrage_opportunities()
    if opportunities:
        print(f"Found {len(opportunities)} arbitrage opportunities")
        print(f"Best opportunity: {opportunities[0]['name']} with {opportunities[0]['roi']:.1%} ROI")
    else:
        print("No arbitrage opportunities found")
    
    # Test most profitable items
    profitable_items = market_analyzer.get_most_profitable_items(3)
    print("Most profitable items:")
    for item in profitable_items:
        print(f"  - {item['name']}: {item['price']/10000:.2f}g (Volume: {item['volume']:.1f})")
    
    print("Market Analyzer test completed")

def test_farming_optimizer():
    """Test farming optimizer functionality."""
    print("\n=== Testing Farming Optimizer ===")
    
    config = get_test_config()
    market_analyzer = MarketAnalyzer(config)
    farming_optimizer = FarmingOptimizer(config, market_analyzer)
    
    # Add test resource nodes
    nodes = generate_test_resource_nodes()
    for node in nodes:
        farming_optimizer.add_node(node)
    
    # Test route calculation
    profession_skills = {"herbalism": 300, "mining": 150}
    route = farming_optimizer.calculate_optimal_route(
        zone="Un'Goro Crater",
        character_level=60,
        profession_skills=profession_skills
    )
    
    if route:
        print(f"Found route in {route.zone} with {len(route.nodes)} nodes")
        print(f"Estimated gold per hour: {route.gold_per_hour/10000:.2f}g")
        print(f"Resource types: {', '.join(route.resource_types)}")
    else:
        print("No suitable route found")
    
    # Test best routes
    best_routes = farming_optimizer.get_best_farming_routes(
        character_level=60,
        profession_skills=profession_skills
    )
    
    print(f"Found {len(best_routes)} optimal farming routes")
    if best_routes:
        print("Best routes:")
        for i, route in enumerate(best_routes, 1):
            print(f"  {i}. {route.zone}: {route.gold_per_hour/10000:.2f}g/hr with {len(route.nodes)} nodes")
    
    print("Farming Optimizer test completed")

def test_crafting_manager():
    """Test crafting manager functionality."""
    print("\n=== Testing Crafting Manager ===")
    
    config = get_test_config()
    market_analyzer = MarketAnalyzer(config)
    
    # Add test auction data first
    auction_data = generate_test_auction_data()
    market_analyzer.update_prices(auction_data)
    
    crafting_manager = CraftingManager(config, market_analyzer)
    
    # Add test recipes
    recipes = generate_test_recipes()
    for recipe in recipes:
        crafting_manager.add_recipe(recipe)
    
    # Test getting profitable recipes
    profitable_recipes = crafting_manager.get_profitable_recipes(
        profession="alchemy",
        min_skill=0,
        min_profit=0
    )
    
    print(f"Found {len(profitable_recipes)} profitable alchemy recipes")
    if profitable_recipes:
        print("Most profitable recipes:")
        for i, recipe in enumerate(profitable_recipes, 1):
            print(f"  {i}. {recipe.name}: Profit {recipe.profit_per_craft/10000:.2f}g ({recipe.profit_margin:.1f}% margin)")
    
    # Test optimal crafting batches
    crafting_batches = crafting_manager.get_optimal_crafting_batches(
        gold_budget=1000 * 10000,  # 1000 gold
        profession="alchemy",
        skill_level=300
    )
    
    print("\nOptimal crafting strategies:")
    for strategy, batches in crafting_batches.items():
        print(f"  {strategy} strategy:")
        for recipe, quantity in batches:
            print(f"    - Craft {quantity}x {recipe.name}")
    
    # Test material sourcing
    if profitable_recipes:
        best_recipe = profitable_recipes[0]
        sourcing = crafting_manager.analyze_material_sourcing(best_recipe.recipe_id)
        
        print(f"\nMaterial sourcing for {best_recipe.name}:")
        for material, options in sourcing["sourcing_options"].items():
            best_option = options[0]
            print(f"  {material}: Best option is to {best_option['method']} - {best_option['details']}")
    
    print("Crafting Manager test completed")

def test_inventory_manager():
    """Test inventory manager functionality."""
    print("\n=== Testing Inventory Manager ===")
    
    config = get_test_config()
    market_analyzer = MarketAnalyzer(config)
    inventory_manager = InventoryManager(config, market_analyzer)
    
    # Add test items
    items = generate_test_inventory()
    for item in items:
        inventory_manager.add_item(item)
    
    # Test inventory optimization
    recommendations = inventory_manager.optimize_inventory()
    
    print("Inventory optimization recommendations:")
    for location, items in recommendations.items():
        if items:
            print(f"  {location.value} ({len(items)} items):")
            for item in items[:3]:  # Show top 3
                print(f"    - {item.name} x{item.quantity}")
            if len(items) > 3:
                print(f"    - ... and {len(items) - 3} more items")
    
    # Test vendor vs AH analysis
    analysis = inventory_manager.vendor_vs_ah_analysis()
    
    print("\nVendor vs AH analysis:")
    print(f"  Vendor: {len(analysis['vendor_items'])} items worth {analysis['vendor_total']/10000:.2f}g")
    print(f"  Auction House: {len(analysis['auction_items'])} items worth {analysis['auction_net']/10000:.2f}g (after fees)")
    
    # Test inventory value analysis
    value_analysis = inventory_manager.analyze_inventory_value()
    
    print("\nInventory value analysis:")
    print(f"  Total value: {value_analysis['total_value']/10000:.2f}g")
    print("  Most valuable items:")
    for item in value_analysis["most_valuable_items"][:3]:
        print(f"    - {item['name']}: {item['value']/10000:.2f}g")
    
    print("Inventory Manager test completed")

def test_economic_manager():
    """Test economic manager functionality."""
    print("\n=== Testing Economic Manager ===")
    
    config = get_test_config()
    economic_manager = EconomicManager(config)
    
    # Update with test auction data
    auction_data = generate_test_auction_data()
    economic_manager.update_market_data(auction_data)
    
    # Add resource nodes
    nodes = generate_test_resource_nodes()
    for node in nodes:
        node_data = {
            "node_id": node.node_id,
            "resource_type": node.resource_type,
            "name": node.name,
            "x": node.x,
            "y": node.y,
            "zone": node.zone,
            "level_req": node.level_req,
            "skill_req": node.skill_req
        }
        economic_manager.add_resource_node(node_data)
    
    # Add recipes
    recipes = generate_test_recipes()
    for recipe in recipes:
        recipe_data = {
            "recipe_id": recipe.recipe_id,
            "name": recipe.name,
            "profession": recipe.profession,
            "skill_req": recipe.skill_req,
            "result_item_id": recipe.result_item_id,
            "result_item_name": recipe.result_item_name,
            "result_quantity": recipe.result_quantity,
            "crafting_cost": recipe.crafting_cost,
            "materials": [
                {
                    "item_id": material.item_id,
                    "name": material.name,
                    "quantity": material.quantity
                }
                for material in recipe.materials
            ]
        }
        economic_manager.add_recipe(recipe_data)
    
    # Update inventory
    inventory_data = []
    for item in generate_test_inventory():
        item_data = {
            "item_id": item.item_id,
            "name": item.name,
            "category": item.category.value,
            "stack_size": item.stack_size,
            "quantity": item.quantity,
            "unit_value": item.unit_value,
            "is_soulbound": item.is_soulbound,
            "item_level": item.item_level,
            "required_level": getattr(item, "required_level", 1),
            "is_quest_item": getattr(item, "is_quest_item", False)
        }
        inventory_data.append(item_data)
    economic_manager.scan_inventory(inventory_data)
    
    # Test getting optimal farming routes
    farming_routes = economic_manager.get_optimal_farming_routes()
    print("Best farming routes:")
    for i, route in enumerate(farming_routes, 1):
        print(f"  {i}. {route['zone']}: {route['gold_per_hour']}g/hr with {route['nodes']} nodes")
    
    # Test getting profitable crafts
    profitable_crafts = economic_manager.get_profitable_crafts(profession="alchemy")
    print("\nProfitable alchemy crafts:")
    for i, craft in enumerate(profitable_crafts, 1):
        print(f"  {i}. {craft['name']}: Profit {craft['profit']/10000:.2f}g ({craft['profit_margin']})")
    
    # Test inventory optimization
    inventory_recommendations = economic_manager.optimize_inventory()
    print("\nInventory recommendations:")
    print(f"  Vendor: {len(inventory_recommendations['vendor'])} items worth {inventory_recommendations['estimated_vendor_value']/10000:.2f}g")
    print(f"  Auction: {len(inventory_recommendations['auction'])} items worth {inventory_recommendations['estimated_auction_value']/10000:.2f}g")
    
    # Test gold earning plan
    gold_plan = economic_manager.make_gold_earning_plan()
    print("\nGold earning recommendations:")
    for i, recommendation in enumerate(gold_plan.get("recommendations", []), 1):
        print(f"  {i}. ({recommendation['priority']}): {recommendation['details']}")
    
    print("Economic Manager test completed")

def main():
    """Run all tests."""
    # Ensure data directory exists
    os.makedirs("data/economic", exist_ok=True)
    
    test_market_analyzer()
    test_farming_optimizer()
    test_crafting_manager()
    test_inventory_manager()
    test_economic_manager()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()