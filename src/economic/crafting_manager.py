"""
Crafting intelligence system for profitability analysis and material sourcing.
"""
from typing import Dict, List, Tuple, Optional, Set
import logging
import json
from dataclasses import dataclass, field

from ..utils.config import Config
from .market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class Material:
    """Represents a crafting material."""
    item_id: int
    name: str
    quantity: int
    unit_price: int = 0
    
    @property
    def total_cost(self) -> int:
        """Calculate total cost of materials."""
        return self.unit_price * self.quantity

@dataclass
class Recipe:
    """Represents a crafting recipe."""
    recipe_id: int
    name: str
    profession: str
    skill_req: int
    result_item_id: int
    result_item_name: str
    result_quantity: int = 1
    materials: List[Material] = field(default_factory=list)
    crafting_cost: int = 0  # Additional costs like vendor materials
    cooldown: int = 0  # Cooldown in seconds, 0 means no cooldown
    
    @property
    def total_material_cost(self) -> int:
        """Calculate total material cost."""
        return sum(material.total_cost for material in self.materials)
    
    @property
    def total_cost(self) -> int:
        """Calculate total cost including crafting fee."""
        return self.total_material_cost + self.crafting_cost
    
    @property
    def profit_per_craft(self) -> int:
        """Calculate profit per craft operation."""
        return (self.result_quantity * self.market_value) - self.total_cost
    
    @property
    def profit_margin(self) -> float:
        """Calculate profit margin as percentage."""
        if self.total_cost == 0:
            return 0.0
        return (self.profit_per_craft / self.total_cost) * 100
    
    # Market value gets updated by CraftingManager
    market_value: int = 0

class CraftingManager:
    """
    Manages crafting decisions and profitability analysis.
    
    This class handles:
    - Profitability calculation for crafted items
    - Material sourcing optimization
    - Recipe unlocking prioritization
    - Crafting batch size optimization
    """
    
    def __init__(self, config: Config, market_analyzer: MarketAnalyzer):
        """
        Initialize the CraftingManager.
        
        Args:
            config: Application configuration
            market_analyzer: Market analyzer for price data
        """
        self.config = config
        self.market_analyzer = market_analyzer
        self.recipes: Dict[int, Recipe] = {}
        self.recipes_by_profession: Dict[str, List[int]] = {}
        self.recipes_by_result: Dict[int, List[int]] = {}
        
        self.recipes_data_file = self.config.get(
            "paths.economic.recipes_data", 
            "data/economic/recipes.json"
        )
        self.load_recipes_data()
        
    def load_recipes_data(self) -> None:
        """Load existing recipe data from disk."""
        try:
            with open(self.recipes_data_file, 'r') as f:
                data = json.load(f)
                
            for recipe_data in data:
                recipe_id = recipe_data["recipe_id"]
                
                # Create materials list
                materials = []
                for mat in recipe_data.get("materials", []):
                    material = Material(
                        item_id=mat["item_id"],
                        name=mat["name"],
                        quantity=mat["quantity"],
                        unit_price=mat.get("unit_price", 0)
                    )
                    materials.append(material)
                
                # Create recipe
                recipe = Recipe(
                    recipe_id=recipe_id,
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
                
                self.recipes[recipe_id] = recipe
                
                # Index by profession
                if recipe.profession not in self.recipes_by_profession:
                    self.recipes_by_profession[recipe.profession] = []
                self.recipes_by_profession[recipe.profession].append(recipe_id)
                
                # Index by result item
                if recipe.result_item_id not in self.recipes_by_result:
                    self.recipes_by_result[recipe.result_item_id] = []
                self.recipes_by_result[recipe.result_item_id].append(recipe_id)
                
            logger.info(f"Loaded {len(self.recipes)} recipes across {len(self.recipes_by_profession)} professions")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No previous recipe data found or file is corrupted")
            self.recipes = {}
            self.recipes_by_profession = {}
            self.recipes_by_result = {}
            
    def save_recipes_data(self) -> None:
        """Save recipe data to disk."""
        data = []
        
        for recipe_id, recipe in self.recipes.items():
            recipe_data = {
                "recipe_id": recipe.recipe_id,
                "name": recipe.name,
                "profession": recipe.profession,
                "skill_req": recipe.skill_req,
                "result_item_id": recipe.result_item_id,
                "result_item_name": recipe.result_item_name,
                "result_quantity": recipe.result_quantity,
                "crafting_cost": recipe.crafting_cost,
                "cooldown": recipe.cooldown,
                "materials": []
            }
            
            for material in recipe.materials:
                mat_data = {
                    "item_id": material.item_id,
                    "name": material.name,
                    "quantity": material.quantity,
                    "unit_price": material.unit_price
                }
                recipe_data["materials"].append(mat_data)
                
            data.append(recipe_data)
            
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.recipes_data_file), exist_ok=True)
            
            with open(self.recipes_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data for {len(self.recipes)} recipes")
        except Exception as e:
            logger.error(f"Failed to save recipe data: {e}")
            
    def add_recipe(self, recipe: Recipe) -> None:
        """
        Add a new recipe or update an existing one.
        
        Args:
            recipe: The recipe to add
        """
        self.recipes[recipe.recipe_id] = recipe
        
        # Index by profession
        if recipe.profession not in self.recipes_by_profession:
            self.recipes_by_profession[recipe.profession] = []
        if recipe.recipe_id not in self.recipes_by_profession[recipe.profession]:
            self.recipes_by_profession[recipe.profession].append(recipe.recipe_id)
            
        # Index by result item
        if recipe.result_item_id not in self.recipes_by_result:
            self.recipes_by_result[recipe.result_item_id] = []
        if recipe.recipe_id not in self.recipes_by_result[recipe.result_item_id]:
            self.recipes_by_result[recipe.result_item_id].append(recipe.recipe_id)
            
        # Update material prices and market value
        self._update_recipe_prices(recipe)
        
        # Save updated data
        self.save_recipes_data()
        
    def update_market_prices(self) -> None:
        """Update all recipe prices based on current market data."""
        for recipe_id in self.recipes:
            self._update_recipe_prices(self.recipes[recipe_id])
            
        self.save_recipes_data()
        
    def _update_recipe_prices(self, recipe: Recipe) -> None:
        """
        Update a recipe's material costs and result market value.
        
        Args:
            recipe: Recipe to update
        """
        # Update material prices
        for material in recipe.materials:
            # Look up current price in market analyzer
            # In a real implementation, we would query actual market prices
            # For now, simulate with some reasonable values
            if hasattr(self.market_analyzer, "price_history") and material.item_id in self.market_analyzer.price_history:
                prices = sorted(self.market_analyzer.price_history[material.item_id], 
                               key=lambda x: x.timestamp)
                if prices:
                    material.unit_price = prices[-1].min_price
            else:
                # Simulate price data if not available
                # In a real implementation, this would come from market data
                base_material_prices = {
                    "herb": 20,
                    "ore": 30,
                    "leather": 15,
                    "cloth": 10,
                    "gem": 100,
                    "metal bar": 50,
                    "potion": 40,
                    "elixir": 60
                }
                
                # Try to guess material type from name
                for mat_type, price in base_material_prices.items():
                    if mat_type in material.name.lower():
                        material.unit_price = price
                        break
                else:
                    # Default if no match
                    material.unit_price = 25
        
        # Update result item market value
        if hasattr(self.market_analyzer, "price_history") and recipe.result_item_id in self.market_analyzer.price_history:
            prices = sorted(self.market_analyzer.price_history[recipe.result_item_id], 
                           key=lambda x: x.timestamp)
            if prices:
                recipe.market_value = prices[-1].min_price
        else:
            # Simulate market value if not available
            # In a real implementation, this would come from market data
            recipe.market_value = recipe.total_cost * 1.2  # Default 20% markup
            
    def get_profitable_recipes(
        self, 
        profession: Optional[str] = None,
        min_skill: int = 0,
        min_profit: int = 0,
        min_profit_margin: float = 0.0,
        max_recipes: int = 20
    ) -> List[Recipe]:
        """
        Get the most profitable recipes for a profession.
        
        Args:
            profession: Profession to filter by (None for all)
            min_skill: Minimum skill level
            min_profit: Minimum profit per craft
            min_profit_margin: Minimum profit margin (percentage)
            max_recipes: Maximum number of recipes to return
            
        Returns:
            List of profitable recipes
        """
        # Update prices first
        self.update_market_prices()
        
        # Filter recipes
        filtered_recipes = []
        
        recipe_ids = []
        if profession and profession in self.recipes_by_profession:
            recipe_ids = self.recipes_by_profession[profession]
        else:
            # All recipes
            recipe_ids = list(self.recipes.keys())
            
        for recipe_id in recipe_ids:
            recipe = self.recipes[recipe_id]
            
            # Filter by skill
            if recipe.skill_req < min_skill:
                continue
                
            # Filter by profit
            if recipe.profit_per_craft < min_profit:
                continue
                
            # Filter by profit margin
            if recipe.profit_margin < min_profit_margin:
                continue
                
            filtered_recipes.append(recipe)
            
        # Sort by profit per craft
        filtered_recipes.sort(key=lambda r: r.profit_per_craft, reverse=True)
        
        return filtered_recipes[:max_recipes]
        
    def get_optimal_crafting_batches(
        self, 
        gold_budget: int,
        profession: Optional[str] = None,
        skill_level: int = 0
    ) -> Dict[str, List[Tuple[Recipe, int]]]:
        """
        Calculate optimal crafting batches based on budget.
        
        Args:
            gold_budget: Budget in gold
            profession: Profession to filter by (None for all)
            skill_level: Character's skill level
            
        Returns:
            Dictionary mapping strategies to recipe batches
        """
        # Update prices
        self.update_market_prices()
        
        # Get applicable recipes
        profitable_recipes = self.get_profitable_recipes(
            profession=profession,
            min_skill=0,  # Include all recipes for now, we'll filter later
            min_profit=1,  # At least some profit
            max_recipes=100  # Get a good number to work with
        )
        
        # Filter by skill level
        available_recipes = [r for r in profitable_recipes if r.skill_req <= skill_level]
        
        if not available_recipes:
            return {"no_profitable_recipes": []}
            
        # Different strategies
        strategies = {}
        
        # 1. Max Profit Strategy - prioritize highest profit per craft
        max_profit_batches = []
        remaining_budget = gold_budget
        
        # Sort by profit per craft
        for recipe in sorted(available_recipes, key=lambda r: r.profit_per_craft, reverse=True):
            # Skip if we can't afford even one
            if recipe.total_cost > remaining_budget:
                continue
                
            # Calculate how many we can craft
            max_crafts = remaining_budget // recipe.total_cost
            
            if max_crafts > 0:
                max_profit_batches.append((recipe, max_crafts))
                remaining_budget -= max_crafts * recipe.total_cost
                
        strategies["max_profit"] = max_profit_batches
        
        # 2. Balanced Strategy - mix of high profit and high margin
        balanced_batches = []
        remaining_budget = gold_budget
        
        # Create a score based on profit and margin
        scored_recipes = []
        for recipe in available_recipes:
            profit_score = recipe.profit_per_craft / max(r.profit_per_craft for r in available_recipes)
            margin_score = recipe.profit_margin / 100  # Convert percentage to 0-1
            
            # Combined score (50% profit, 50% margin)
            score = (profit_score * 0.5) + (margin_score * 0.5)
            scored_recipes.append((recipe, score))
            
        # Sort by score
        scored_recipes.sort(key=lambda x: x[1], reverse=True)
        
        for recipe, _ in scored_recipes:
            if recipe.total_cost > remaining_budget:
                continue
                
            max_crafts = remaining_budget // recipe.total_cost
            
            # Limit to a reasonable batch size to diversify
            actual_crafts = min(max_crafts, 5)
            
            if actual_crafts > 0:
                balanced_batches.append((recipe, actual_crafts))
                remaining_budget -= actual_crafts * recipe.total_cost
                
        strategies["balanced"] = balanced_batches
        
        # 3. Quickest Return Strategy - prioritize recipes with lower total cost
        quick_return_batches = []
        remaining_budget = gold_budget
        
        # Sort by total cost (ascending) but must have positive profit
        for recipe in sorted(
            [r for r in available_recipes if r.profit_per_craft > 0], 
            key=lambda r: r.total_cost
        ):
            if recipe.total_cost > remaining_budget:
                continue
                
            max_crafts = remaining_budget // recipe.total_cost
            
            if max_crafts > 0:
                quick_return_batches.append((recipe, max_crafts))
                remaining_budget -= max_crafts * recipe.total_cost
                
        strategies["quick_return"] = quick_return_batches
        
        return strategies
        
    def get_skill_leveling_path(self, profession: str, current_skill: int, target_skill: int) -> List[Tuple[Recipe, int]]:
        """
        Calculate optimal recipe sequence for leveling a profession.
        
        Args:
            profession: Profession to level
            current_skill: Current skill level
            target_skill: Target skill level
            
        Returns:
            List of (recipe, count) tuples for leveling
        """
        if profession not in self.recipes_by_profession:
            return []
            
        # Update prices
        self.update_market_prices()
        
        # Get all recipes for this profession
        profession_recipes = [self.recipes[rid] for rid in self.recipes_by_profession[profession]]
        
        # Sort by skill requirement
        profession_recipes.sort(key=lambda r: r.skill_req)
        
        # Build leveling path
        leveling_path = []
        current = current_skill
        
        while current < target_skill:
            # Find recipes in the appropriate skill range
            valid_recipes = [
                r for r in profession_recipes 
                if r.skill_req <= current and r.skill_req > current - 10
            ]
            
            if not valid_recipes:
                # No recipes in the right range, get the lowest skill req above current
                higher_recipes = [r for r in profession_recipes if r.skill_req > current]
                if not higher_recipes:
                    break  # No higher recipes available
                    
                valid_recipes = [min(higher_recipes, key=lambda r: r.skill_req)]
                
            # Pick the cheapest recipe
            cheapest = min(valid_recipes, key=lambda r: r.total_cost)
            
            # Estimate skill points - in a real implementation this would be more complex
            # based on recipe color (orange, yellow, green, gray)
            skill_range = cheapest.skill_req + 10 - current
            crafts_needed = max(1, (skill_range + 4) // 5)  # Roughly 1 point per 5 crafts
            
            leveling_path.append((cheapest, crafts_needed))
            current += skill_range
            
        return leveling_path
        
    def analyze_material_sourcing(self, recipe_id: int) -> Dict:
        """
        Analyze the best ways to source materials for a recipe.
        
        Args:
            recipe_id: Recipe to analyze
            
        Returns:
            Dictionary with sourcing analysis
        """
        if recipe_id not in self.recipes:
            return {"error": "Recipe not found"}
            
        recipe = self.recipes[recipe_id]
        
        # Update prices
        self._update_recipe_prices(recipe)
        
        sourcing_options = {}
        
        for material in recipe.materials:
            # Options for this material
            options = []
            
            # 1. Buy directly option
            buy_option = {
                "method": "buy",
                "cost": material.unit_price * material.quantity,
                "details": f"Buy {material.quantity}x {material.name} @ {material.unit_price/10000:.2f}g each"
            }
            options.append(buy_option)
            
            # 2. Craft option (if material can be crafted)
            if material.item_id in self.recipes_by_result:
                craft_recipes = [self.recipes[rid] for rid in self.recipes_by_result[material.item_id]]
                
                # Update prices for these recipes
                for craft_recipe in craft_recipes:
                    self._update_recipe_prices(craft_recipe)
                    
                # Find cheapest crafting option
                if craft_recipes:
                    cheapest_recipe = min(craft_recipes, key=lambda r: r.total_cost / r.result_quantity)
                    
                    # Calculate how many crafts needed
                    crafts_needed = (material.quantity + cheapest_recipe.result_quantity - 1) // cheapest_recipe.result_quantity
                    
                    craft_option = {
                        "method": "craft",
                        "cost": cheapest_recipe.total_cost * crafts_needed,
                        "details": f"Craft {crafts_needed}x {cheapest_recipe.name} @ {cheapest_recipe.total_cost/10000:.2f}g each"
                    }
                    options.append(craft_option)
            
            # 3. Farm option
            # In a real implementation, this would use FarmingOptimizer to calculate farming efficiency
            # For now, we'll just estimate based on material type
            farm_time_per_unit = {
                "herb": 30,  # seconds per unit
                "ore": 45,
                "leather": 20,
                "cloth": 15
            }
            
            material_type = None
            for typ in farm_time_per_unit:
                if typ in material.name.lower():
                    material_type = typ
                    break
                    
            if material_type:
                time_seconds = farm_time_per_unit[material_type] * material.quantity
                gold_per_hour = 3600 * (material.unit_price / farm_time_per_unit[material_type])
                
                farm_option = {
                    "method": "farm",
                    "cost": 0,  # Free but takes time
                    "time_minutes": time_seconds / 60,
                    "gold_per_hour": gold_per_hour / 10000,  # Convert to gold
                    "details": f"Farm {material.quantity}x {material.name} (~{time_seconds/60:.1f} minutes)"
                }
                options.append(farm_option)
            
            # Sort options by cost
            options.sort(key=lambda o: o["cost"])
            
            sourcing_options[material.name] = options
            
        # Calculate total costs
        total_buy_cost = sum(opt[0]["cost"] for opt in sourcing_options.values())
        
        return {
            "recipe": recipe.name,
            "total_materials_cost": recipe.total_material_cost,
            "profit_per_craft": recipe.profit_per_craft,
            "profit_margin": recipe.profit_margin,
            "sourcing_options": sourcing_options,
            "buy_all_cost": total_buy_cost
        }