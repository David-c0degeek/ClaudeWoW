# Economic Intelligence System

ClaudeWoW's Economic Intelligence System provides sophisticated market analysis and decision-making capabilities to optimize gold earning and resource management.

## Overview

The Economic Intelligence System is designed to help players make informed economic decisions within the game, including optimizing farming routes, crafting profitable items, and managing inventory efficiently.

## Components

### Market Analyzer

The Market Analyzer (`market_analyzer.py`) is responsible for analyzing auction house data to identify trends and opportunities.

#### Features:
- Price trend detection and prediction
- Arbitrage opportunity identification
- Profitable item identification
- Historical price tracking
- Market volume analysis

#### Usage:
```python
from src.economic.market_analyzer import MarketAnalyzer

# Initialize with config
market_analyzer = MarketAnalyzer(config)

# Update with auction data
market_analyzer.update_prices(auction_data)

# Get price trend for an item
trend = market_analyzer.get_item_trend(item_id)

# Find arbitrage opportunities
opportunities = market_analyzer.find_arbitrage_opportunities(min_roi=0.15)

# Get most profitable items
profitable_items = market_analyzer.get_most_profitable_items(n=10)
```

### Farming Optimizer

The Farming Optimizer (`farming_optimizer.py`) manages resource nodes and calculates optimal farming routes.

#### Features:
- Resource node mapping and tracking
- Optimal farming route calculation
- Time/gold efficiency modeling
- Dynamic route adjustment based on competition
- Multi-resource type optimization

#### Usage:
```python
from src.economic.farming_optimizer import FarmingOptimizer

# Initialize with config and market analyzer
farming_optimizer = FarmingOptimizer(config, market_analyzer)

# Add or update a resource node
farming_optimizer.add_node(node)

# Record node harvesting
farming_optimizer.record_node_visit(node_id)

# Calculate optimal farming route
route = farming_optimizer.calculate_optimal_route(
    zone="Un'Goro Crater",
    resource_types=["herb", "ore"],
    character_level=60,
    profession_skills={"herbalism": 300, "mining": 275}
)

# Get best farming routes
best_routes = farming_optimizer.get_best_farming_routes(
    character_level=60,
    profession_skills={"herbalism": 300, "mining": 275},
    top_n=3
)
```

### Crafting Manager

The Crafting Manager (`crafting_manager.py`) evaluates crafting profitability and manages recipes.

#### Features:
- Profitability calculation for crafted items
- Material sourcing optimization
- Recipe unlocking prioritization
- Crafting batch size optimization
- Strategy-based crafting recommendations

#### Usage:
```python
from src.economic.crafting_manager import CraftingManager

# Initialize with config and market analyzer
crafting_manager = CraftingManager(config, market_analyzer)

# Add a recipe
crafting_manager.add_recipe(recipe)

# Get profitable recipes
profitable_recipes = crafting_manager.get_profitable_recipes(
    profession="alchemy",
    min_skill=250,
    min_profit=5000,
    min_profit_margin=15.0
)

# Get optimal crafting batches based on budget
batches = crafting_manager.get_optimal_crafting_batches(
    gold_budget=1000 * 10000,  # 1000 gold
    profession="alchemy",
    skill_level=300
)

# Analyze material sourcing options
sourcing = crafting_manager.analyze_material_sourcing(recipe_id)
```

### Inventory Manager

The Inventory Manager (`inventory_manager.py`) handles inventory organization and decision-making.

#### Features:
- Value-based inventory prioritization
- Bag space optimization
- Vendor vs. AH decision making
- Bank storage organization
- Inventory value analysis

#### Usage:
```python
from src.economic.inventory_manager import InventoryManager

# Initialize with config and market analyzer
inventory_manager = InventoryManager(config, market_analyzer)

# Add an item to inventory
inventory_manager.add_item(item)

# Remove an item from inventory
inventory_manager.remove_item(item_id, quantity=1)

# Optimize inventory
recommendations = inventory_manager.optimize_inventory()

# Prioritize bag space
items_to_remove = inventory_manager.prioritize_bag_space(free_slots_needed=5)

# Get bank organization plan
organization_plan = inventory_manager.get_bank_organization_plan()

# Analyze inventory value
value_analysis = inventory_manager.analyze_inventory_value()

# Compare vendor vs AH selling
sell_analysis = inventory_manager.vendor_vs_ah_analysis()
```

### Economic Manager

The Economic Manager (`economic_manager.py`) coordinates the economic intelligence components and provides integrated functionality.

#### Features:
- Centralized economic decision-making
- Coordinated market, farming, crafting, and inventory management
- Gold-earning plan generation
- Comprehensive economic insights

#### Usage:
```python
from src.economic.economic_manager import EconomicManager

# Initialize with config
economic_manager = EconomicManager(config)

# Update character information
economic_manager.update_character_info(
    level=60,
    gold=10000 * 10000,  # 10,000 gold
    profession_skills={"herbalism": 300, "alchemy": 300}
)

# Update market data
economic_manager.update_market_data(auction_data)

# Scan inventory
economic_manager.scan_inventory(inventory_data)

# Get optimal farming routes
farming_routes = economic_manager.get_optimal_farming_routes()

# Get profitable crafts
profitable_crafts = economic_manager.get_profitable_crafts(profession="alchemy")

# Get crafting shopping list
shopping_list = economic_manager.get_crafting_shopping_list([recipe_id], [quantity])

# Optimize inventory
inventory_recommendations = economic_manager.optimize_inventory()

# Get comprehensive gold earning plan
gold_plan = economic_manager.make_gold_earning_plan()
```

## Integration with Agent

The Economic Intelligence System is integrated with the main agent through the EconomicManager instance. The agent can access economic insights and make informed decisions based on market conditions, farming opportunities, crafting profitability, and inventory status.

## Demo Script

To see the Economic Intelligence System in action, run the demo script:

```
python economic_demo.py
```

This will demonstrate key economic intelligence features with simulated data.

## Data Storage

The Economic Intelligence System persists data in JSON files in the `data/economic/` directory:

- `price_data.json` - Historical auction house price data
- `resource_nodes.json` - Resource node locations and information
- `recipes.json` - Crafting recipe data
- `inventory.json` - Inventory item data
- `containers.json` - Container (bags, bank) data

## Future Improvements

Planned improvements to the Economic Intelligence System include:

1. Machine learning for price prediction using time series analysis
2. Cross-server arbitrage detection and recommendations
3. Advanced market manipulation strategies
4. Economic risk assessment and diversification
5. Crafting queue optimization for efficiency