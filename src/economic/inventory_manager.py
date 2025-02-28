"""
Inventory management system for optimizing bag space and storage decisions.
"""
from typing import Dict, List, Tuple, Optional, Set
import logging
import json
import heapq
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..utils.config import Config
from .market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)

class ItemCategory(Enum):
    """Categories for inventory items."""
    CONSUMABLE = "consumable"
    MATERIAL = "material"
    EQUIPMENT = "equipment"
    QUEST = "quest"
    CONTAINER = "container"
    TRADE_GOODS = "trade_goods"
    RECIPE = "recipe"
    COIN = "coin"
    KEY = "key"
    JUNK = "junk"
    MISC = "misc"

class StorageLocation(Enum):
    """Possible storage locations for items."""
    BAG = "bag"
    BANK = "bank"
    GUILD_BANK = "guild_bank"
    MAIL = "mail"
    ALT = "alt"
    AUCTION = "auction"
    VENDOR = "vendor"  # For items to sell to vendor
    DISCARD = "discard"  # For items to discard

@dataclass
class InventoryItem:
    """Represents an item in inventory."""
    item_id: int
    name: str
    category: ItemCategory
    stack_size: int
    quantity: int
    unit_value: int = 0  # Market value per unit
    is_soulbound: bool = False
    item_level: int = 1
    required_level: int = 1
    is_unique: bool = False
    is_quest_item: bool = False
    expiration: Optional[datetime] = None  # For items with expiration
    location: StorageLocation = StorageLocation.BAG
    slot: Optional[int] = None  # Bag and slot position
    
    @property
    def total_value(self) -> int:
        """Total market value of the item stack."""
        return self.unit_value * self.quantity
    
    @property
    def slots_used(self) -> int:
        """Number of inventory slots used by this item."""
        return (self.quantity + self.stack_size - 1) // self.stack_size
    
    @property
    def value_per_slot(self) -> float:
        """Average value per inventory slot used."""
        if self.slots_used == 0:
            return 0
        return self.total_value / self.slots_used
    
    @property
    def is_expired(self) -> bool:
        """Check if item is expired."""
        if not self.expiration:
            return False
        return datetime.now() > self.expiration

@dataclass
class StorageContainer:
    """Represents a storage container (bag, bank tab, etc.)."""
    container_id: int
    name: str
    slot_count: int
    used_slots: int = 0
    items: List[InventoryItem] = field(default_factory=list)
    
    @property
    def free_slots(self) -> int:
        """Number of free slots in the container."""
        return self.slot_count - self.used_slots
    
    @property
    def total_value(self) -> int:
        """Total value of all items in the container."""
        return sum(item.total_value for item in self.items)

class InventoryManager:
    """
    Manages inventory organization and decision making.
    
    This class handles:
    - Value-based inventory prioritization
    - Bag space optimization
    - Vendor vs. AH decision making
    - Bank storage organization
    """
    
    def __init__(self, config: Config, market_analyzer: MarketAnalyzer):
        """
        Initialize the InventoryManager.
        
        Args:
            config: Application configuration
            market_analyzer: Market analyzer for price data
        """
        self.config = config
        self.market_analyzer = market_analyzer
        self.items: Dict[int, InventoryItem] = {}  # All items by item_id
        self.containers: Dict[int, StorageContainer] = {}  # All containers by container_id
        
        # Group items by location
        self.items_by_location: Dict[StorageLocation, List[int]] = {loc: [] for loc in StorageLocation}
        
        # Group items by category
        self.items_by_category: Dict[ItemCategory, List[int]] = {cat: [] for cat in ItemCategory}
        
        self.inventory_data_file = self.config.get(
            "paths.economic.inventory_data", 
            "data/economic/inventory.json"
        )
        self.container_data_file = self.config.get(
            "paths.economic.container_data", 
            "data/economic/containers.json"
        )
        
        # Value thresholds - configurable
        self.vendor_threshold = self.config.get("inventory.vendor_threshold", 100)  # items worth less than this sell to vendor
        self.ah_threshold = self.config.get("inventory.ah_threshold", 500)  # items worth more than this sell on AH
        self.keep_threshold = self.config.get("inventory.keep_threshold", 5000)  # valuable materials to keep
        
        self.load_inventory_data()
        self.load_container_data()
        
    def load_inventory_data(self) -> None:
        """Load existing inventory data from disk."""
        try:
            with open(self.inventory_data_file, 'r') as f:
                data = json.load(f)
                
            for item_data in data:
                item_id = item_data["item_id"]
                
                # Create item
                item = InventoryItem(
                    item_id=item_id,
                    name=item_data["name"],
                    category=ItemCategory(item_data["category"]),
                    stack_size=item_data["stack_size"],
                    quantity=item_data["quantity"],
                    unit_value=item_data.get("unit_value", 0),
                    is_soulbound=item_data.get("is_soulbound", False),
                    item_level=item_data.get("item_level", 1),
                    required_level=item_data.get("required_level", 1),
                    is_unique=item_data.get("is_unique", False),
                    is_quest_item=item_data.get("is_quest_item", False),
                    expiration=datetime.fromisoformat(item_data["expiration"]) 
                        if "expiration" in item_data else None,
                    location=StorageLocation(item_data.get("location", "bag")),
                    slot=item_data.get("slot")
                )
                
                self.items[item_id] = item
                
                # Index by location
                self.items_by_location[item.location].append(item_id)
                
                # Index by category
                self.items_by_category[item.category].append(item_id)
                
            logger.info(f"Loaded {len(self.items)} inventory items")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No previous inventory data found or file is corrupted")
            self.items = {}
            self.items_by_location = {loc: [] for loc in StorageLocation}
            self.items_by_category = {cat: [] for cat in ItemCategory}
            
    def load_container_data(self) -> None:
        """Load existing container data from disk."""
        try:
            with open(self.container_data_file, 'r') as f:
                data = json.load(f)
                
            for container_data in data:
                container_id = container_data["container_id"]
                
                # Create container
                container = StorageContainer(
                    container_id=container_id,
                    name=container_data["name"],
                    slot_count=container_data["slot_count"],
                    used_slots=container_data["used_slots"]
                )
                
                # Add items to container
                if "items" in container_data:
                    for item_id in container_data["items"]:
                        if item_id in self.items:
                            container.items.append(self.items[item_id])
                
                self.containers[container_id] = container
                
            logger.info(f"Loaded {len(self.containers)} containers")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No previous container data found or file is corrupted")
            self.containers = {}
            
    def save_inventory_data(self) -> None:
        """Save inventory data to disk."""
        data = []
        
        for item_id, item in self.items.items():
            item_data = {
                "item_id": item.item_id,
                "name": item.name,
                "category": item.category.value,
                "stack_size": item.stack_size,
                "quantity": item.quantity,
                "unit_value": item.unit_value,
                "is_soulbound": item.is_soulbound,
                "item_level": item.item_level,
                "required_level": item.required_level,
                "is_unique": item.is_unique,
                "is_quest_item": item.is_quest_item,
                "location": item.location.value,
                "slot": item.slot
            }
            
            if item.expiration:
                item_data["expiration"] = item.expiration.isoformat()
                
            data.append(item_data)
            
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.inventory_data_file), exist_ok=True)
            
            with open(self.inventory_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data for {len(self.items)} inventory items")
        except Exception as e:
            logger.error(f"Failed to save inventory data: {e}")
            
    def save_container_data(self) -> None:
        """Save container data to disk."""
        data = []
        
        for container_id, container in self.containers.items():
            container_data = {
                "container_id": container.container_id,
                "name": container.name,
                "slot_count": container.slot_count,
                "used_slots": container.used_slots,
                "items": [item.item_id for item in container.items]
            }
            
            data.append(container_data)
            
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.container_data_file), exist_ok=True)
            
            with open(self.container_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data for {len(self.containers)} containers")
        except Exception as e:
            logger.error(f"Failed to save container data: {e}")
            
    def update_item_values(self) -> None:
        """Update item values based on current market data."""
        for item_id, item in self.items.items():
            # Look up current price in market analyzer
            if hasattr(self.market_analyzer, "price_history") and item_id in self.market_analyzer.price_history:
                prices = sorted(self.market_analyzer.price_history[item_id], 
                                key=lambda x: x.timestamp)
                if prices:
                    item.unit_value = prices[-1].min_price
            
        self.save_inventory_data()
        
    def add_item(self, item: InventoryItem, container_id: Optional[int] = None) -> bool:
        """
        Add a new item to inventory or update an existing one.
        
        Args:
            item: The item to add
            container_id: Optional container to add to
            
        Returns:
            True if successful, False if inventory is full
        """
        # Add to a specific container if specified
        if container_id is not None:
            if container_id not in self.containers:
                logger.error(f"Container {container_id} not found")
                return False
                
            container = self.containers[container_id]
            if container.free_slots < item.slots_used:
                logger.error(f"Not enough free slots in container {container.name}")
                return False
                
            container.items.append(item)
            container.used_slots += item.slots_used
            
        # Add or update item in main inventory
        if item.item_id in self.items:
            # Update existing item
            existing_item = self.items[item.item_id]
            existing_item.quantity += item.quantity
            
            # Update location if needed
            if existing_item.location != item.location:
                # Remove from old location index
                if existing_item.item_id in self.items_by_location[existing_item.location]:
                    self.items_by_location[existing_item.location].remove(existing_item.item_id)
                
                # Update location
                existing_item.location = item.location
                
                # Add to new location index
                self.items_by_location[item.location].append(existing_item.item_id)
        else:
            # Add new item
            self.items[item.item_id] = item
            
            # Add to indexes
            self.items_by_location[item.location].append(item.item_id)
            self.items_by_category[item.category].append(item.item_id)
            
        # Update item value
        if hasattr(self.market_analyzer, "price_history") and item.item_id in self.market_analyzer.price_history:
            prices = sorted(self.market_analyzer.price_history[item.item_id], 
                           key=lambda x: x.timestamp)
            if prices:
                item.unit_value = prices[-1].min_price
                
        # Save updated data
        self.save_inventory_data()
        if container_id is not None:
            self.save_container_data()
            
        return True
        
    def remove_item(self, item_id: int, quantity: int = 1) -> bool:
        """
        Remove an item from inventory.
        
        Args:
            item_id: ID of the item to remove
            quantity: Quantity to remove
            
        Returns:
            True if successful, False if item not found or insufficient quantity
        """
        if item_id not in self.items:
            logger.error(f"Item {item_id} not found in inventory")
            return False
            
        item = self.items[item_id]
        
        if item.quantity < quantity:
            logger.error(f"Not enough {item.name} in inventory. Have {item.quantity}, need {quantity}")
            return False
            
        # Update quantity
        item.quantity -= quantity
        
        # Remove item if quantity is 0
        if item.quantity == 0:
            # Remove from indexes
            if item_id in self.items_by_location[item.location]:
                self.items_by_location[item.location].remove(item_id)
                
            if item_id in self.items_by_category[item.category]:
                self.items_by_category[item.category].remove(item_id)
                
            # Remove from container if in one
            for container in self.containers.values():
                for i, cont_item in enumerate(container.items):
                    if cont_item.item_id == item_id:
                        container.items.pop(i)
                        container.used_slots -= item.slots_used
                        break
                        
            # Remove from main inventory
            del self.items[item_id]
            
        # Save updated data
        self.save_inventory_data()
        self.save_container_data()
        
        return True
        
    def optimize_inventory(self) -> Dict[StorageLocation, List[InventoryItem]]:
        """
        Optimize inventory organization based on item value and purpose.
        
        Returns:
            Dictionary mapping recommended actions to item lists
        """
        # Update item values first
        self.update_item_values()
        
        recommendations = {loc: [] for loc in StorageLocation}
        
        # Process each item in bags
        for item_id in list(self.items_by_location[StorageLocation.BAG]):
            item = self.items[item_id]
            
            # Skip soulbound items
            if item.is_soulbound:
                continue
                
            # Skip quest items
            if item.is_quest_item:
                continue
                
            # Process based on category and value
            if item.category == ItemCategory.JUNK:
                # Junk always goes to vendor
                recommendations[StorageLocation.VENDOR].append(item)
                
            elif item.category == ItemCategory.MATERIAL or item.category == ItemCategory.TRADE_GOODS:
                # Low value materials to vendor
                if item.unit_value < self.vendor_threshold:
                    recommendations[StorageLocation.VENDOR].append(item)
                    
                # Medium value materials to AH
                elif self.vendor_threshold <= item.unit_value < self.keep_threshold:
                    recommendations[StorageLocation.AUCTION].append(item)
                    
                # High value materials to bank for crafting
                else:
                    recommendations[StorageLocation.BANK].append(item)
                    
            elif item.category == ItemCategory.EQUIPMENT:
                # Equipment decisions based on item level and character level
                # For now, just use value
                if item.unit_value < self.ah_threshold:
                    recommendations[StorageLocation.VENDOR].append(item)
                else:
                    recommendations[StorageLocation.AUCTION].append(item)
                    
            elif item.category == ItemCategory.CONSUMABLE:
                # Keep consumables unless they're expired
                if item.is_expired:
                    recommendations[StorageLocation.DISCARD].append(item)
                    
            # Recipes always to AH if not learned
            elif item.category == ItemCategory.RECIPE:
                recommendations[StorageLocation.AUCTION].append(item)
                
        return recommendations
        
    def prioritize_bag_space(self, free_slots_needed: int = 0) -> List[InventoryItem]:
        """
        Prioritize items to clear bag space based on value per slot.
        
        Args:
            free_slots_needed: Number of free slots needed
            
        Returns:
            List of items to remove
        """
        if free_slots_needed <= 0:
            return []
            
        # Calculate current free slots
        total_bag_slots = sum(
            container.slot_count 
            for container in self.containers.values() 
            if container.name.startswith("Bag")
        )
        
        used_bag_slots = sum(
            container.used_slots 
            for container in self.containers.values() 
            if container.name.startswith("Bag")
        )
        
        current_free_slots = total_bag_slots - used_bag_slots
        
        # If we already have enough free slots, return empty list
        if current_free_slots >= free_slots_needed:
            return []
            
        slots_to_free = free_slots_needed - current_free_slots
        
        # Get all items in bags
        bag_items = []
        for container in self.containers.values():
            if container.name.startswith("Bag"):
                bag_items.extend(container.items)
                
        # Sort items by increasing value per slot
        bag_items.sort(key=lambda item: item.value_per_slot)
        
        # Select items to remove
        items_to_remove = []
        freed_slots = 0
        
        for item in bag_items:
            # Skip quest items and soulbound items
            if item.is_quest_item or item.is_soulbound:
                continue
                
            items_to_remove.append(item)
            freed_slots += item.slots_used
            
            if freed_slots >= slots_to_free:
                break
                
        return items_to_remove
        
    def get_bank_organization_plan(self) -> Dict[str, List[InventoryItem]]:
        """
        Generate a plan for organizing bank storage by item category.
        
        Returns:
            Dictionary mapping bank tab names to item lists
        """
        # Define bank organization by category
        bank_tabs = {
            "Materials": [ItemCategory.MATERIAL, ItemCategory.TRADE_GOODS],
            "Equipment": [ItemCategory.EQUIPMENT],
            "Consumables": [ItemCategory.CONSUMABLE],
            "Recipes": [ItemCategory.RECIPE],
            "Miscellaneous": [ItemCategory.MISC, ItemCategory.CONTAINER, ItemCategory.KEY]
        }
        
        # Get all items in bank
        bank_items = [
            self.items[item_id] 
            for item_id in self.items_by_location[StorageLocation.BANK]
        ]
        
        # Organize by tab
        organization_plan = {tab_name: [] for tab_name in bank_tabs}
        
        for item in bank_items:
            for tab_name, categories in bank_tabs.items():
                if item.category in categories:
                    organization_plan[tab_name].append(item)
                    break
            else:
                # If no match, put in Miscellaneous
                organization_plan["Miscellaneous"].append(item)
                
        return organization_plan
        
    def analyze_inventory_value(self) -> Dict:
        """
        Analyze total inventory value by category and location.
        
        Returns:
            Dictionary with inventory value analysis
        """
        # Update item values first
        self.update_item_values()
        
        analysis = {
            "total_value": 0,
            "by_category": {},
            "by_location": {},
            "most_valuable_items": []
        }
        
        # Calculate total value
        for item in self.items.values():
            analysis["total_value"] += item.total_value
            
            # By category
            category = item.category.value
            if category not in analysis["by_category"]:
                analysis["by_category"][category] = 0
            analysis["by_category"][category] += item.total_value
            
            # By location
            location = item.location.value
            if location not in analysis["by_location"]:
                analysis["by_location"][location] = 0
            analysis["by_location"][location] += item.total_value
            
        # Find most valuable items
        all_items = list(self.items.values())
        all_items.sort(key=lambda item: item.total_value, reverse=True)
        
        analysis["most_valuable_items"] = [
            {
                "name": item.name,
                "value": item.total_value,
                "category": item.category.value,
                "location": item.location.value
            }
            for item in all_items[:10]  # Top 10 items
        ]
        
        return analysis
        
    def vendor_vs_ah_analysis(self) -> Dict:
        """
        Analyze whether items should be sold to vendor or auction house.
        
        Returns:
            Dictionary with vendor vs AH analysis
        """
        # Update item values first
        self.update_item_values()
        
        analysis = {
            "vendor_items": [],
            "auction_items": [],
            "vendor_total": 0,
            "auction_total": 0,
            "auction_fee": 0,
            "auction_net": 0
        }
        
        # Process each item in bags
        for item_id in self.items_by_location[StorageLocation.BAG]:
            item = self.items[item_id]
            
            # Skip soulbound/quest items
            if item.is_soulbound or item.is_quest_item:
                continue
                
            # Calculate vendor value (assuming 25% of market value)
            vendor_value = int(item.total_value * 0.25)
            
            # Calculate AH value after fees (assuming 5% AH cut)
            ah_fee = int(item.total_value * 0.05)
            ah_net = item.total_value - ah_fee
            
            # Decide where to sell
            if ah_net > vendor_value and item.unit_value >= self.ah_threshold:
                analysis["auction_items"].append({
                    "name": item.name,
                    "quantity": item.quantity,
                    "market_value": item.total_value,
                    "ah_fee": ah_fee,
                    "ah_net": ah_net
                })
                analysis["auction_total"] += item.total_value
                analysis["auction_fee"] += ah_fee
                analysis["auction_net"] += ah_net
            else:
                analysis["vendor_items"].append({
                    "name": item.name,
                    "quantity": item.quantity,
                    "vendor_value": vendor_value
                })
                analysis["vendor_total"] += vendor_value
                
        return analysis