# src/knowledge/game_knowledge.py

import os
import json
import logging
import math
from typing import Dict, List, Tuple, Any, Optional, Union

class GameKnowledge:
    """
    This class manages game knowledge including items, quests, NPCs, zones, etc.
    It loads data from JSON files and provides access methods.
    """
    
    def update(self, game_state):
        """
        Update knowledge with new game state information
        
        Args:
            game_state: Current game state
        """
        # This method will be implemented to update knowledge from game state
        # Currently just a stub to avoid errors
        pass
    
    def __init__(self, config: Dict):
        """
        Initialize GameKnowledge
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.knowledge.game_knowledge")
        self.config = config
        
        # Data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "game_knowledge"
        )
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Data storage
        self.items = {}
        self.quests = {}
        self.npcs = {}
        self.zones = {}
        self.abilities = {}
        self.professions = {}
        self.vendors = {}
        self.flight_paths = {}
        self.instances = {}
        self.emotes = {}
        
        # Load data
        self._load_data()
        
        self.logger.info("GameKnowledge initialized")
    
    def _load_data(self) -> None:
        """
        Load all game knowledge data
        """
        self._load_items()
        self._load_quests()
        self._load_npcs()
        self._load_zones()
        self._load_abilities()
        self._load_professions()
        self._load_vendors()
        self._load_flight_paths()
        self._load_instances()
        self._load_emotes()
    
    def _load_items(self) -> None:
        """
        Load item data
        """
        file_path = os.path.join(self.data_dir, "items.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_items_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.items = json.load(f)
                self.logger.info(f"Loaded {len(self.items)} items")
        except Exception as e:
            self.logger.error(f"Error loading items data: {e}")
            # Create default data on error
            self._create_default_items_data(file_path)
    
    def _load_quests(self) -> None:
        """
        Load quest data
        """
        file_path = os.path.join(self.data_dir, "quests.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_quests_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.quests = json.load(f)
                self.logger.info(f"Loaded {len(self.quests)} quests")
        except Exception as e:
            self.logger.error(f"Error loading quests data: {e}")
            # Create default data on error
            self._create_default_quests_data(file_path)
    
    def _load_npcs(self) -> None:
        """
        Load NPC data
        """
        file_path = os.path.join(self.data_dir, "npcs.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_npcs_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.npcs = json.load(f)
                self.logger.info(f"Loaded {len(self.npcs)} NPCs")
        except Exception as e:
            self.logger.error(f"Error loading NPCs data: {e}")
            # Create default data on error
            self._create_default_npcs_data(file_path)
    
    def _load_zones(self) -> None:
        """
        Load zone data
        """
        file_path = os.path.join(self.data_dir, "zones.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_zones_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.zones = json.load(f)
                self.logger.info(f"Loaded {len(self.zones)} zones")
        except Exception as e:
            self.logger.error(f"Error loading zones data: {e}")
            # Create default data on error
            self._create_default_zones_data(file_path)
    
    def _load_abilities(self) -> None:
        """
        Load ability data
        """
        file_path = os.path.join(self.data_dir, "abilities.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_abilities_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.abilities = json.load(f)
                self.logger.info(f"Loaded abilities for {len(self.abilities)} classes")
        except Exception as e:
            self.logger.error(f"Error loading abilities data: {e}")
            # Create default data on error
            self._create_default_abilities_data(file_path)
    
    def _load_professions(self) -> None:
        """
        Load profession data
        """
        file_path = os.path.join(self.data_dir, "professions.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_professions_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.professions = json.load(f)
                self.logger.info(f"Loaded {len(self.professions)} professions")
        except Exception as e:
            self.logger.error(f"Error loading professions data: {e}")
            # Create default data on error
            self._create_default_professions_data(file_path)
    
    def _load_vendors(self) -> None:
        """
        Load vendor data
        """
        file_path = os.path.join(self.data_dir, "vendors.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_vendors_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.vendors = json.load(f)
                self.logger.info(f"Loaded {len(self.vendors)} vendors")
        except Exception as e:
            self.logger.error(f"Error loading vendors data: {e}")
            # Create default data on error
            self._create_default_vendors_data(file_path)
    
    def _load_flight_paths(self) -> None:
        """
        Load flight path data
        """
        file_path = os.path.join(self.data_dir, "flight_paths.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_flight_paths_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.flight_paths = json.load(f)
                self.logger.info(f"Loaded flight paths for {len(self.flight_paths)} factions")
        except Exception as e:
            self.logger.error(f"Error loading flight paths data: {e}")
            # Create default data on error
            self._create_default_flight_paths_data(file_path)
    
    def _load_instances(self) -> None:
        """
        Load instance data (dungeons/raids)
        """
        file_path = os.path.join(self.data_dir, "instances.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_instances_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.instances = json.load(f)
                self.logger.info(f"Loaded {len(self.instances)} instances")
        except Exception as e:
            self.logger.error(f"Error loading instances data: {e}")
            # Create default data on error
            self._create_default_instances_data(file_path)
    
    def _load_emotes(self) -> None:
        """
        Load emote data
        """
        file_path = os.path.join(self.data_dir, "emotes.json")
        
        # Create default file if it doesn't exist
        if not os.path.exists(file_path):
            self._create_default_emotes_data(file_path)
        
        try:
            with open(file_path, 'r') as f:
                self.emotes = json.load(f)
                self.logger.info(f"Loaded {len(self.emotes)} emotes")
        except Exception as e:
            self.logger.error(f"Error loading emotes data: {e}")
            # Create default data on error
            self._create_default_emotes_data(file_path)
    
    def _create_default_items_data(self, file_path: str) -> None:
        """
        Create default items data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "hearthstone": {
                "id": 6948,
                "name": "Hearthstone",
                "type": "misc",
                "quality": "common",
                "level": 1,
                "bind": "soulbound",
                "unique": True,
                "use": "Returns you to your home location.",
                "cooldown": 3600
            },
            "linen_cloth": {
                "id": 2589,
                "name": "Linen Cloth",
                "type": "trade",
                "quality": "common",
                "level": 1,
                "bind": "none",
                "unique": False,
                "sell_price": 15
            },
            "copper_ore": {
                "id": 2770,
                "name": "Copper Ore",
                "type": "trade",
                "quality": "common",
                "level": 1,
                "bind": "none",
                "unique": False,
                "sell_price": 20
            },
            "light_leather": {
                "id": 2318,
                "name": "Light Leather",
                "type": "trade",
                "quality": "common",
                "level": 1,
                "bind": "none",
                "unique": False,
                "sell_price": 18
            },
            "copper_shortsword": {
                "id": 2504,
                "name": "Worn Shortsword",
                "type": "weapon",
                "subtype": "one-hand sword",
                "quality": "common",
                "level": 1,
                "bind": "none",
                "unique": False,
                "damage": {"min": 2, "max": 4},
                "speed": 1.8,
                "dps": 1.67,
                "stats": {},
                "sell_price": 23
            }
        }
        
        self.items = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default items data at {file_path}")
    
    def _create_default_quests_data(self, file_path: str) -> None:
        """
        Create default quests data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "defias_brotherhood": {
                "id": 155,
                "name": "The Defias Brotherhood",
                "level": 14,
                "min_level": 9,
                "faction": "alliance",
                "zone": "westfall",
                "giver": "Gryan Stoutmantle",
                "objectives": [
                    "Kill 15 Defias Messenger",
                    "Collect the Unsent Letter"
                ],
                "rewards": {
                    "xp": 875,
                    "gold": 25,
                    "items": [
                        {"id": 1006, "name": "Tough Leather Boots", "quality": "uncommon"},
                        {"id": 1007, "name": "Studded Leather Belt", "quality": "uncommon"}
                    ]
                },
                "follow_up": "The Defias Brotherhood (Part 2)"
            },
            "lazy_peons": {
                "id": 5441,
                "name": "Lazy Peons",
                "level": 2,
                "min_level": 1,
                "faction": "horde",
                "zone": "durotar",
                "giver": "Foreman Thazz'ril",
                "objectives": [
                    "Use the Foreman's Blackjack on 5 Lazy Peons"
                ],
                "rewards": {
                    "xp": 250,
                    "gold": 5,
                    "items": [
                        {"id": 6267, "name": "Disciple's Pants", "quality": "common"}
                    ]
                },
                "follow_up": "Thazz'ril's Pick"
            }
        }
        
        self.quests = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default quests data at {file_path}")
    
    def _create_default_npcs_data(self, file_path: str) -> None:
        """
        Create default NPCs data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "hogger": {
                "id": 448,
                "name": "Hogger",
                "level": 11,
                "type": "elite",
                "faction": "enemy",
                "zone": "elwynn_forest",
                "location": {"x": 25.2, "y": 78.5},
                "hp": 500,
                "abilities": ["Rushing Charge", "Eat", "Vicious Bite"],
                "loot": [
                    {"id": 1127, "name": "Chipped Boar Tusk", "drop_rate": 0.25},
                    {"id": 3667, "name": "Ruined Pelt", "drop_rate": 0.15},
                    {"id": 1114, "name": "Tough Jerky", "drop_rate": 0.3}
                ],
                "quest_related": ["Wanted: Hogger"]
            },
            "innkeeper_renee": {
                "id": 6741,
                "name": "Innkeeper Renee",
                "level": 35,
                "type": "normal",
                "faction": "alliance",
                "zone": "stormwind",
                "location": {"x": 52.3, "y": 67.8},
                "services": ["inn", "vendor", "food_drink"],
                "gossip": [
                    "Welcome to the Gilded Rose. May I help you?",
                    "Looking for a place to stay? We have the softest beds in Stormwind!"
                ]
            }
        }
        
        self.npcs = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default NPCs data at {file_path}")
    
    def _create_default_zones_data(self, file_path: str) -> None:
        """
        Create default zones data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "elwynn_forest": {
                "id": 1429,
                "name": "Elwynn Forest",
                "faction": "alliance",
                "level_range": [1, 10],
                "adjacent_zones": ["stormwind", "westfall", "redridge_mountains", "duskwood"],
                "flight_points": ["stormwind"],
                "main_city": "stormwind",
                "notable_npcs": ["Marshal Dughan", "Hogger", "Goldshire Guards"],
                "notable_locations": [
                    {"name": "Goldshire", "type": "town", "coords": {"x": 42.1, "y": 65.9}},
                    {"name": "Northshire Abbey", "type": "quest_hub", "coords": {"x": 45.6, "y": 47.1}},
                    {"name": "Fargodeep Mine", "type": "mine", "coords": {"x": 39.5, "y": 81.7}}
                ]
            },
            "durotar": {
                "id": 1411,
                "name": "Durotar",
                "faction": "horde",
                "level_range": [1, 10],
                "adjacent_zones": ["orgrimmar", "the_barrens"],
                "flight_points": ["orgrimmar", "razor_hill"],
                "main_city": "orgrimmar",
                "notable_npcs": ["Foreman Thazz'ril", "Zuljin Headhunters", "Razormane Quilboars"],
                "notable_locations": [
                    {"name": "Razor Hill", "type": "town", "coords": {"x": 52.1, "y": 43.2}},
                    {"name": "Valley of Trials", "type": "starting_area", "coords": {"x": 43.6, "y": 68.9}},
                    {"name": "Tiragarde Keep", "type": "elite_area", "coords": {"x": 58.7, "y": 57.1}}
                ]
            }
        }
        
        self.zones = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default zones data at {file_path}")
    
    def _create_default_abilities_data(self, file_path: str) -> None:
        """
        Create default abilities data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "warrior": {
                "battle_stance": {
                    "name": "Battle Stance",
                    "level": 1,
                    "type": "stance",
                    "cost": "None",
                    "cooldown": 1.5,
                    "description": "A balanced combat stance.",
                    "effect": "Allows use of Battle stance abilities."
                },
                "heroic_strike": {
                    "name": "Heroic Strike",
                    "level": 1,
                    "type": "rage",
                    "cost": 15,
                    "cooldown": 1.5,
                    "description": "A strong attack that increases damage.",
                    "effect": "Instant. Adds 45 damage to your next attack.",
                    "stance": "battle"
                },
                "charge": {
                    "name": "Charge",
                    "level": 4,
                    "type": "rage",
                    "cost": 0,
                    "cooldown": 15,
                    "description": "Charges an enemy, generates rage and stuns.",
                    "effect": "Generates 9 rage, stuns for 1 sec.",
                    "stance": "battle",
                    "range": "8-25 yards"
                }
            },
            "mage": {
                "fireball": {
                    "name": "Fireball",
                    "level": 1,
                    "type": "mana",
                    "cost": 30,
                    "cooldown": 1.5,
                    "cast_time": 2.5,
                    "description": "Hurls a fiery ball that burns the enemy.",
                    "effect": "Deals 18 to 24 Fire damage, plus 6 over 6 sec.",
                    "range": "30 yards"
                },
                "frostbolt": {
                    "name": "Frostbolt",
                    "level": 4,
                    "type": "mana",
                    "cost": 25,
                    "cooldown": 1.5,
                    "cast_time": 2.0,
                    "description": "Launches a bolt of frost at the enemy.",
                    "effect": "Deals 16 to 18 Frost damage and slows movement by 50% for 5 sec.",
                    "range": "30 yards"
                }
            }
        }
        
        self.abilities = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default abilities data at {file_path}")
    
    def _create_default_professions_data(self, file_path: str) -> None:
        """
        Create default professions data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "alchemy": {
                "name": "Alchemy",
                "type": "crafting",
                "required_skill": "none",
                "trainer_locations": {
                    "alliance": [
                        {"zone": "stormwind", "npc": "Lilyssia Nightbreeze", "coords": {"x": 55.8, "y": 86.1}},
                        {"zone": "ironforge", "npc": "Tally Berryfizz", "coords": {"x": 68.3, "y": 42.5}}
                    ],
                    "horde": [
                        {"zone": "orgrimmar", "npc": "Yelmak", "coords": {"x": 56.2, "y": 33.8}},
                        {"zone": "undercity", "npc": "Doctor Herbert Halsey", "coords": {"x": 47.5, "y": 73.9}}
                    ]
                },
                "recipes": {
                    "minor_healing_potion": {
                        "name": "Minor Healing Potion",
                        "skill_required": 1,
                        "reagents": [
                            {"item": "peacebloom", "count": 1},
                            {"item": "silverleaf", "count": 1},
                            {"item": "empty_vial", "count": 1}
                        ],
                        "result": {"item": "minor_healing_potion", "count": 1}
                    }
                }
            },
            "mining": {
                "name": "Mining",
                "type": "gathering",
                "required_skill": "none",
                "trainer_locations": {
                    "alliance": [
                        {"zone": "stormwind", "npc": "Gelman Stonehand", "coords": {"x": 59.7, "y": 37.5}},
                        {"zone": "ironforge", "npc": "Geofram Bouldertoe", "coords": {"x": 50.9, "y": 26.5}}
                    ],
                    "horde": [
                        {"zone": "orgrimmar", "npc": "Makaru", "coords": {"x": 73.1, "y": 26.1}},
                        {"zone": "thunder_bluff", "npc": "Brek Stonehoof", "coords": {"x": 36.2, "y": 59.8}}
                    ]
                },
                "nodes": {
                    "copper_vein": {
                        "name": "Copper Vein",
                        "skill_required": 1,
                        "yields": [
                            {"item": "copper_ore", "chance": 1.0},
                            {"item": "rough_stone", "chance": 0.5},
                            {"item": "tigerseye", "chance": 0.05}
                        ],
                        "typical_zones": ["elwynn_forest", "dun_morogh", "durotar", "tirisfal_glades"]
                    }
                }
            }
        }
        
        self.professions = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default professions data at {file_path}")
    
    def _create_default_vendors_data(self, file_path: str) -> None:
        """
        Create default vendors data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "elwynn_general_store": {
                "name": "Elwynn General Store",
                "zone": "elwynn_forest",
                "subzone": "goldshire",
                "faction": "alliance",
                "npc": "Andrew Krighton",
                "location": {"x": 41.5, "y": 65.8},
                "items": [
                    {"id": 2320, "name": "Coarse Thread", "price": 10, "stock": "unlimited"},
                    {"id": 2321, "name": "Fine Thread", "price": 100, "stock": "unlimited"},
                    {"id": 4291, "name": "Silken Thread", "price": 500, "stock": "unlimited"},
                    {"id": 2678, "name": "Mild Spices", "price": 10, "stock": "unlimited"},
                    {"id": 4470, "name": "Simple Wood", "price": 5, "stock": "unlimited"},
                    {"id": 4498, "name": "Brown Leather Satchel", "price": 25, "stock": "unlimited"}
                ]
            },
            "razor_hill_supply": {
                "name": "Razor Hill Supply",
                "zone": "durotar",
                "subzone": "razor_hill",
                "faction": "horde",
                "npc": "Uhgar",
                "location": {"x": 52.5, "y": 41.6},
                "items": [
                    {"id": 2320, "name": "Coarse Thread", "price": 10, "stock": "unlimited"},
                    {"id": 2321, "name": "Fine Thread", "price": 100, "stock": "unlimited"},
                    {"id": 4291, "name": "Silken Thread", "price": 500, "stock": "unlimited"},
                    {"id": 2678, "name": "Mild Spices", "price": 10, "stock": "unlimited"},
                    {"id": 4470, "name": "Simple Wood", "price": 5, "stock": "unlimited"},
                    {"id": 4499, "name": "Brown Leather Satchel", "price": 25, "stock": "unlimited"}
                ]
            }
        }
        
        self.vendors = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default vendors data at {file_path}")
    
    def _create_default_flight_paths_data(self, file_path: str) -> None:
        """
        Create default flight paths data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "alliance": {
                "stormwind": {
                    "name": "Stormwind",
                    "location": {"x": 66.3, "y": 62.1, "zone": "elwynn_forest"},
                    "connections": [
                        {"to": "sentinel_hill", "cost": 50, "time": 60},
                        {"to": "lakeshire", "cost": 50, "time": 75},
                        {"to": "darkshire", "cost": 70, "time": 90},
                        {"to": "ironforge", "cost": 110, "time": 120}
                    ]
                },
                "ironforge": {
                    "name": "Ironforge",
                    "location": {"x": 55.5, "y": 47.7, "zone": "dun_morogh"},
                    "connections": [
                        {"to": "stormwind", "cost": 110, "time": 120},
                        {"to": "thelsamar", "cost": 50, "time": 60},
                        {"to": "menethil_harbor", "cost": 75, "time": 90}
                    ]
                }
            },
            "horde": {
                "orgrimmar": {
                    "name": "Orgrimmar",
                    "location": {"x": 45.1, "y": 63.9, "zone": "durotar"},
                    "connections": [
                        {"to": "crossroads", "cost": 50, "time": 60},
                        {"to": "thunder_bluff", "cost": 110, "time": 180},
                        {"to": "undercity", "cost": 225, "time": 300}
                    ]
                },
                "thunder_bluff": {
                    "name": "Thunder Bluff",
                    "location": {"x": 46.8, "y": 49.9, "zone": "mulgore"},
                    "connections": [
                        {"to": "orgrimmar", "cost": 110, "time": 180},
                        {"to": "crossroads", "cost": 70, "time": 120},
                        {"to": "sun_rock_retreat", "cost": 60, "time": 90}
                    ]
                }
            }
        }
        
        self.flight_paths = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default flight paths data at {file_path}")
    
    def get_item_info(self, item_name: str) -> Optional[Dict]:
        """
        Get information about an item
        
        Args:
            item_name: Name or ID of the item
            
        Returns:
            Dict: Item information or None if not found
        """
        # Normalize item name for key lookup
        item_key = item_name.lower().replace(" ", "_")
        
        # Check if item exists directly
        if item_key in self.items:
            return self.items[item_key]
        
        # Try to find by ID if item_name is numeric
        if str(item_name).isdigit():
            item_id = int(item_name)
            for key, data in self.items.items():
                if data.get("id") == item_id:
                    return data
        
        # Try to find by full name (case insensitive)
        for key, data in self.items.items():
            if data.get("name", "").lower() == item_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in self.items.items():
            if item_name.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple items match '{item_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_quest_info(self, quest_name: str) -> Optional[Dict]:
        """
        Get information about a quest
        
        Args:
            quest_name: Name or ID of the quest
            
        Returns:
            Dict: Quest information or None if not found
        """
        # Normalize quest name for key lookup
        quest_key = quest_name.lower().replace(" ", "_")
        
        # Check if quest exists directly
        if quest_key in self.quests:
            return self.quests[quest_key]
        
        # Try to find by ID if quest_name is numeric
        if str(quest_name).isdigit():
            quest_id = int(quest_name)
            for key, data in self.quests.items():
                if data.get("id") == quest_id:
                    return data
        
        # Try to find by full name (case insensitive)
        for key, data in self.quests.items():
            if data.get("name", "").lower() == quest_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in self.quests.items():
            if quest_name.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple quests match '{quest_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_npc_info(self, npc_name: str) -> Optional[Dict]:
        """
        Get information about an NPC
        
        Args:
            npc_name: Name or ID of the NPC
            
        Returns:
            Dict: NPC information or None if not found
        """
        # Normalize NPC name for key lookup
        npc_key = npc_name.lower().replace(" ", "_")
        
        # Check if NPC exists directly
        if npc_key in self.npcs:
            return self.npcs[npc_key]
        
        # Try to find by ID if npc_name is numeric
        if str(npc_name).isdigit():
            npc_id = int(npc_name)
            for key, data in self.npcs.items():
                if data.get("id") == npc_id:
                    return data
        
        # Try to find by full name (case insensitive)
        for key, data in self.npcs.items():
            if data.get("name", "").lower() == npc_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in self.npcs.items():
            if npc_name.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple NPCs match '{npc_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_zone_info(self, zone_name: str) -> Optional[Dict]:
        """
        Get information about a zone
        
        Args:
            zone_name: Name or ID of the zone
            
        Returns:
            Dict: Zone information or None if not found
        """
        # Normalize zone name for key lookup
        zone_key = zone_name.lower().replace(" ", "_")
        
        # Check if zone exists directly
        if zone_key in self.zones:
            return self.zones[zone_key]
        
        # Try to find by ID if zone_name is numeric
        if str(zone_name).isdigit():
            zone_id = int(zone_name)
            for key, data in self.zones.items():
                if data.get("id") == zone_id:
                    return data
        
        # Try to find by full name (case insensitive)
        for key, data in self.zones.items():
            if data.get("name", "").lower() == zone_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in self.zones.items():
            if zone_name.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple zones match '{zone_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_ability_info(self, class_name: str, ability_name: str) -> Optional[Dict]:
        """
        Get information about a class ability
        
        Args:
            class_name: Class name
            ability_name: Ability name
            
        Returns:
            Dict: Ability information or None if not found
        """
        class_key = class_name.lower()
        ability_key = ability_name.lower().replace(" ", "_")
        
        if class_key not in self.abilities:
            return None
        
        class_abilities = self.abilities[class_key]
        
        # Check if ability exists directly
        if ability_key in class_abilities:
            return class_abilities[ability_key]
        
        # Try to find by full name (case insensitive)
        for key, data in class_abilities.items():
            if data.get("name", "").lower() == ability_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in class_abilities.items():
            if ability_name.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple abilities match '{ability_name}' for class '{class_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_profession_info(self, profession_name: str) -> Optional[Dict]:
        """
        Get information about a profession
        
        Args:
            profession_name: Profession name
            
        Returns:
            Dict: Profession information or None if not found
        """
        # Normalize profession name for key lookup
        profession_key = profession_name.lower().replace(" ", "_")
        
        # Check if profession exists directly
        if profession_key in self.professions:
            return self.professions[profession_key]
        
        # Try to find by full name (case insensitive)
        for key, data in self.professions.items():
            if data.get("name", "").lower() == profession_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in self.professions.items():
            if profession_name.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple professions match '{profession_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_vendor_info(self, vendor_name: str) -> Optional[Dict]:
        """
        Get information about a vendor
        
        Args:
            vendor_name: Vendor name
            
        Returns:
            Dict: Vendor information or None if not found
        """
        # Normalize vendor name for key lookup
        vendor_key = vendor_name.lower().replace(" ", "_")
        
        # Check if vendor exists directly
        if vendor_key in self.vendors:
            return self.vendors[vendor_key]
        
        # Try to find by full name (case insensitive)
        for key, data in self.vendors.items():
            if data.get("name", "").lower() == vendor_name.lower():
                return data
        
        # Try to find by NPC name
        for key, data in self.vendors.items():
            if data.get("npc", "").lower() == vendor_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in self.vendors.items():
            if vendor_name.lower() in data.get("name", "").lower():
                matches.append(data)
            elif vendor_name.lower() in data.get("npc", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple vendors match '{vendor_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_flight_path_info(self, faction: str, location: str) -> Optional[Dict]:
        """
        Get information about a flight path
        
        Args:
            faction: Alliance or Horde
            location: Flight path location name
            
        Returns:
            Dict: Flight path information or None if not found
        """
        # Normalize inputs
        faction_key = faction.lower()
        location_key = location.lower().replace(" ", "_")
        
        if faction_key not in self.flight_paths:
            return None
        
        faction_fps = self.flight_paths[faction_key]
        
        # Check if flight path exists directly
        if location_key in faction_fps:
            return faction_fps[location_key]
        
        # Try to find by full name (case insensitive)
        for key, data in faction_fps.items():
            if data.get("name", "").lower() == location.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in faction_fps.items():
            if location.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple flight paths match '{location}' for faction '{faction}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def calc_flight_path(self, faction: str, start: str, end: str) -> Optional[List[Dict]]:
        """
        Calculate the optimal flight path between two locations
        
        Args:
            faction: Alliance or Horde
            start: Starting flight point
            end: Ending flight point
            
        Returns:
            List[Dict]: List of flight segments or None if no path found
        """
        faction_key = faction.lower()
        
        if faction_key not in self.flight_paths:
            return None
        
        faction_fps = self.flight_paths[faction_key]
        
        # Find the start and end flight points
        start_fp = None
        for key, data in faction_fps.items():
            if data.get("name", "").lower() == start.lower() or key.lower() == start.lower():
                start_fp = key
                break
        
        end_fp = None
        for key, data in faction_fps.items():
            if data.get("name", "").lower() == end.lower() or key.lower() == end.lower():
                end_fp = key
                break
        
        if not start_fp or not end_fp:
            return None
        
        # Use a simple Dijkstra's algorithm for pathfinding
        # Initialize distances
        distances = {fp: float('infinity') for fp in faction_fps}
        distances[start_fp] = 0
        previous = {fp: None for fp in faction_fps}
        visited = set()
        
        while visited != set(faction_fps):
            current = min([fp for fp in faction_fps if fp not in visited], key=lambda fp: distances[fp])
            
            if current == end_fp:
                break
                
            visited.add(current)
            
            # Check connections
            for connection in faction_fps[current].get("connections", []):
                to_fp = connection.get("to")
                if to_fp in faction_fps:
                    cost = connection.get("cost", 0)
                    if distances[current] + cost < distances[to_fp]:
                        distances[to_fp] = distances[current] + cost
                        previous[to_fp] = current
        
        # Construct path
        if distances[end_fp] == float('infinity'):
            return None  # No path found
        
        path = []
        current = end_fp
        while current != start_fp:
            prev = previous[current]
            # Find connection details
            for conn in faction_fps[prev].get("connections", []):
                if conn.get("to") == current:
                    path.append({
                        "from": prev,
                        "to": current,
                        "cost": conn.get("cost", 0),
                        "time": conn.get("time", 0)
                    })
                    break
            current = prev
        
        # Reverse to get start to end order
        path.reverse()
        
        return path
    
    def calc_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate the distance between two 2D positions
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            float: Distance between positions
        """
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        
    def get_path(self, start: Tuple[float, float], end: Tuple[float, float], zone: str = "") -> Optional[List[Dict]]:
        """
        Get a predefined path between two positions
        
        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            zone: Current zone name
            
        Returns:
            Optional[List[Dict]]: List of waypoint actions or None if no predefined path
        """
        self.logger.debug(f"Searching for path in zone {zone} from {start} to {end}")
        
        # This is a placeholder implementation
        # In a full implementation, we would search a path database
        
        # For now, just return a direct path
        return None
    
    def get_unexplored_areas(self, zone: str, explored_areas: set) -> List[Dict]:
        """
        Get unexplored areas in a zone
        
        Args:
            zone: Zone name
            explored_areas: Set of already explored area identifiers
            
        Returns:
            List[Dict]: List of unexplored areas
        """
        # This is a placeholder implementation
        
        # In a full implementation, we would check a database of known areas
        # against the explored_areas set
        
        return []
        
    def get_quest_turnin(self, quest_title: str) -> Optional[Dict]:
        """
        Get information about an NPC that can accept a completed quest
        
        Args:
            quest_title: Title of the quest
            
        Returns:
            Optional[Dict]: Quest turn-in NPC information or None if not found
        """
        # This is a placeholder implementation
        
        # In a real implementation, this would check the quest database
        # to find the NPC that accepts the completed quest
        
        # Try to get quest data from known quests
        quest_info = self.get_quest_info(quest_title)
        
        if quest_info and "giver" in quest_info:
            # For simple quests, the turn-in NPC is often the same as the giver
            giver_name = quest_info["giver"]
            
            # Look up the NPC info
            npc_info = self.get_npc_info(giver_name)
            
            if npc_info:
                return {
                    "id": npc_info.get("id", 0),
                    "name": npc_info.get("name", giver_name),
                    "position": npc_info.get("location", {"x": 0, "y": 0}),
                    "zone": npc_info.get("zone", "")
                }
        
        return None
        
    def get_quest_giver(self, quest_title: str) -> Optional[Dict]:
        """
        Get information about the NPC that gives a quest
        
        Args:
            quest_title: Title of the quest
            
        Returns:
            Optional[Dict]: Quest giver NPC information or None if not found
        """
        # This is a placeholder implementation
        
        # Try to get quest data from known quests
        quest_info = self.get_quest_info(quest_title)
        
        if quest_info and "giver" in quest_info:
            giver_name = quest_info["giver"]
            
            # Look up the NPC info
            npc_info = self.get_npc_info(giver_name)
            
            if npc_info:
                return {
                    "id": npc_info.get("id", 0),
                    "name": npc_info.get("name", giver_name),
                    "position": npc_info.get("location", {"x": 0, "y": 0}),
                    "zone": npc_info.get("zone", "")
                }
        
        return None
    
    def get_quest_objective_info(self, quest_title: str, objective_name: str) -> Optional[Dict]:
        """
        Get information about a quest objective
        
        Args:
            quest_title: Title of the quest
            objective_name: Name of the objective
            
        Returns:
            Optional[Dict]: Objective information or None if not found
        """
        # This is a placeholder implementation
        
        # In a real implementation, this would look up detailed objective info
        # from a quest database
        
        return {
            "type": "unknown",
            "description": objective_name,
            "location": {"x": 0, "y": 0}  # Default location
        }
    
    def _create_default_instances_data(self, file_path: str) -> None:
        """
        Create default instances data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "deadmines": {
                "name": "Deadmines",
                "level_range": [15, 25],
                "faction": "both",
                "bosses": [
                    {
                        "name": "Rhahk'Zor",
                        "strategy": "Simple first boss. Tank and spank."
                    },
                    {
                        "name": "Sneed",
                        "strategy": "Kill the Shredder first, then Sneed will come out."
                    },
                    {
                        "name": "Gilnid",
                        "strategy": "Kill the adds first, then focus on the boss."
                    },
                    {
                        "name": "Mr. Smite",
                        "strategy": "He stuns at health thresholds and switches weapons. Save cooldowns for after stuns."
                    },
                    {
                        "name": "Captain Greenskin",
                        "strategy": "Tank adds away from the group, focus down the boss."
                    },
                    {
                        "name": "Edwin VanCleef",
                        "strategy": "He summons adds at low health. Save cooldowns for this phase."
                    }
                ],
                "tips": [
                    "Watch for patrols in the foundry area.",
                    "Be careful of the goblin engineers, they can call for help.",
                    "The shortcut through the side tunnel can save time.",
                    "After Mr. Smite, you can jump down to skip trash."
                ]
            },
            "wailing_caverns": {
                "name": "Wailing Caverns",
                "level_range": [15, 25],
                "faction": "both",
                "bosses": [
                    {
                        "name": "Lady Anacondra",
                        "strategy": "Interrupt her Sleep spell. Tank and spank otherwise."
                    },
                    {
                        "name": "Lord Cobrahn",
                        "strategy": "He will transform into a snake at low health. Increased damage, so save defensives."
                    },
                    {
                        "name": "Kresh",
                        "strategy": "Optional boss. Simple tank and spank fight in the water."
                    },
                    {
                        "name": "Lord Pythas",
                        "strategy": "Interrupt his healing spell. Otherwise tank and spank."
                    },
                    {
                        "name": "Skum",
                        "strategy": "Optional boss. Fights in water. Can charge, so stay close."
                    },
                    {
                        "name": "Lord Serpentis",
                        "strategy": "Focuses random targets. Healers be aware."
                    },
                    {
                        "name": "Verdan the Everliving",
                        "strategy": "Final boss. Can knock players back and stun. Tank against a wall."
                    }
                ],
                "tips": [
                    "The dungeon is a maze. Follow the marked path or you'll get lost.",
                    "There are four Fanglord bosses that must be killed before awakening Naralex.",
                    "Deviate Faerie Dragons can put everyone to sleep. Kill them quickly.",
                    "After all bosses, talk to the Disciple to start the final event."
                ]
            }
        }
        
        self.instances = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default instances data at {file_path}")

    def _create_default_emotes_data(self, file_path: str) -> None:
        """
        Create default emotes data file
        
        Args:
            file_path: Path to save the file
        """
        default_data = {
            "wave": {"text": "You wave.", "text_target": "You wave at %s.", "social_impact": "friendly"},
            "smile": {"text": "You smile.", "text_target": "You smile at %s.", "social_impact": "friendly"},
            "laugh": {"text": "You laugh.", "text_target": "You laugh at %s.", "social_impact": "friendly"},
            "thank": {"text": "You thank everyone.", "text_target": "You thank %s.", "social_impact": "friendly"},
            "cheer": {"text": "You cheer!", "text_target": "You cheer at %s!", "social_impact": "friendly"},
            "dance": {"text": "You dance.", "text_target": "You dance with %s.", "social_impact": "friendly"},
            "greet": {"text": "You greet everyone warmly.", "text_target": "You greet %s warmly.", "social_impact": "friendly"},
            "bow": {"text": "You bow.", "text_target": "You bow before %s.", "social_impact": "respectful"},
            "applaud": {"text": "You applaud.", "text_target": "You applaud at %s.", "social_impact": "friendly"},
            "salute": {"text": "You salute.", "text_target": "You salute %s.", "social_impact": "respectful"}
        }
        
        self.emotes = default_data
        
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=4)
        
        self.logger.info(f"Created default emotes data at {file_path}") 

    def get_instance_info(self, instance_name: str) -> Optional[Dict]:
        """
        Get information about a dungeon/raid instance
        
        Args:
            instance_name: Instance name
            
        Returns:
            Dict: Instance information or None if not found
        """
        # Normalize instance name for key lookup
        instance_key = instance_name.lower().replace(" ", "_")
        
        # Check if instance exists directly
        if instance_key in self.instances:
            return self.instances[instance_key]
        
        # Try to find by full name (case insensitive)
        for key, data in self.instances.items():
            if data.get("name", "").lower() == instance_name.lower():
                return data
        
        # Try to find partial match
        matches = []
        for key, data in self.instances.items():
            if instance_name.lower() in data.get("name", "").lower():
                matches.append(data)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple instances match '{instance_name}': {[m.get('name') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None
    
    def get_emote_info(self, emote_name: str) -> Optional[Dict]:
        """
        Get information about an emote
        
        Args:
            emote_name: Emote name or command
            
        Returns:
            Dict: Emote information or None if not found
        """
        # Normalize emote name for key lookup
        emote_key = emote_name.lower().replace(" ", "_")
        
        # Strip leading slash if present
        if emote_key.startswith("/"):
            emote_key = emote_key[1:]
        
        # Check if emote exists directly
        if emote_key in self.emotes:
            return self.emotes[emote_key]
        
        # Try to find partial match
        matches = []
        for key, data in self.emotes.items():
            if emote_key in key:
                matches.append({**data, "command": key})
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            self.logger.warning(f"Multiple emotes match '{emote_name}': {[m.get('command') for m in matches]}")
            return matches[0]  # Return first match with a warning
            
        return None