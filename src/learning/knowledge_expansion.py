"""
Knowledge Expansion System

This module handles the autonomous expansion of the agent's knowledge base
by discovering, validating, and storing new information about the game world.
"""

import logging
import os
import pickle
import json
import time
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

# Setup module-level logger
logger = logging.getLogger("wow_ai.learning.knowledge_expansion")

@dataclass
class KnowledgeEntry:
    """Represents a single piece of knowledge"""
    key: str
    value: Any
    confidence: float = 0.5  # 0.0 to 1.0
    source: str = "observation"  # observation, inference, documentation
    timestamp: float = field(default_factory=time.time)
    times_observed: int = 1
    related_keys: List[str] = field(default_factory=list)


class KnowledgeCategory:
    """Represents a category of knowledge (NPCs, items, quests, etc.)"""
    
    def __init__(self, name: str):
        """
        Initialize a knowledge category
        
        Args:
            name: The name of the category
        """
        self.name = name
        self.entries: Dict[str, KnowledgeEntry] = {}
        
    def add_entry(self, entry: KnowledgeEntry) -> None:
        """
        Add a knowledge entry to this category
        
        Args:
            entry: The knowledge entry to add
        """
        if entry.key in self.entries:
            existing = self.entries[entry.key]
            
            # Update confidence based on repeated observations
            # More observations = higher confidence
            existing.times_observed += 1
            existing.confidence = min(0.95, existing.confidence + 0.05)
            
            # Update value if the new entry has higher confidence
            if entry.confidence > existing.confidence:
                existing.value = entry.value
                existing.source = entry.source
                
            # Update timestamp if newer
            existing.timestamp = max(existing.timestamp, entry.timestamp)
            
            # Add any new related keys
            for related_key in entry.related_keys:
                if related_key not in existing.related_keys:
                    existing.related_keys.append(related_key)
        else:
            self.entries[entry.key] = entry
        
    def get_entry(self, key: str) -> Optional[KnowledgeEntry]:
        """
        Get a knowledge entry by key
        
        Args:
            key: The key to look up
            
        Returns:
            The knowledge entry or None if not found
        """
        return self.entries.get(key)
    
    def get_all_entries(self) -> List[KnowledgeEntry]:
        """
        Get all entries in this category
        
        Returns:
            List of all knowledge entries
        """
        return list(self.entries.values())
    
    def get_entries_by_confidence(self, min_confidence: float = 0.0) -> List[KnowledgeEntry]:
        """
        Get entries with at least the specified confidence level
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of knowledge entries with sufficient confidence
        """
        return [entry for entry in self.entries.values() if entry.confidence >= min_confidence]


class KnowledgeBase:
    """Knowledge base for storing and retrieving agent knowledge"""
    
    def __init__(self):
        """Initialize the knowledge base"""
        self.categories: Dict[str, KnowledgeCategory] = {}
        
        # Initialize standard categories
        standard_categories = [
            "npcs", "items", "quests", "locations", "skills", 
            "game_mechanics", "enemy_behaviors", "dungeon_tactics",
            "recipes", "resources"
        ]
        
        for category in standard_categories:
            self.categories[category] = KnowledgeCategory(category)
        
    def add_category(self, name: str) -> KnowledgeCategory:
        """
        Add a new knowledge category
        
        Args:
            name: Name of the category to add
            
        Returns:
            The newly created category
        """
        if name not in self.categories:
            self.categories[name] = KnowledgeCategory(name)
        return self.categories[name]
    
    def add_entry(self, category: str, entry: KnowledgeEntry) -> None:
        """
        Add a knowledge entry to a category
        
        Args:
            category: Category to add the entry to
            entry: The knowledge entry to add
        """
        if category not in self.categories:
            self.add_category(category)
        
        self.categories[category].add_entry(entry)
        
    def get_entry(self, category: str, key: str) -> Optional[KnowledgeEntry]:
        """
        Get a knowledge entry by category and key
        
        Args:
            category: The category to look in
            key: The key to look up
            
        Returns:
            The knowledge entry or None if not found
        """
        if category not in self.categories:
            return None
        
        return self.categories[category].get_entry(key)
    
    def search(self, query: str, min_confidence: float = 0.0) -> List[Tuple[str, KnowledgeEntry]]:
        """
        Search for knowledge entries matching a query
        
        Args:
            query: Search query
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (category, entry) tuples matching the query
        """
        results = []
        query = query.lower()
        
        for category_name, category in self.categories.items():
            for entry in category.get_all_entries():
                if (entry.confidence >= min_confidence and 
                    (query in entry.key.lower() or 
                     (isinstance(entry.value, str) and query in entry.value.lower()))):
                    results.append((category_name, entry))
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save knowledge base to disk
        
        Args:
            path: Path to save the knowledge base
        """
        data = {
            category_name: {
                entry.key: {
                    "value": entry.value,
                    "confidence": entry.confidence,
                    "source": entry.source,
                    "timestamp": entry.timestamp,
                    "times_observed": entry.times_observed,
                    "related_keys": entry.related_keys
                }
                for entry in category.get_all_entries()
            }
            for category_name, category in self.categories.items()
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Knowledge base saved to {path}")
    
    def load(self, path: str) -> bool:
        """
        Load knowledge base from disk
        
        Args:
            path: Path to load the knowledge base from
            
        Returns:
            Success status
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # Clear existing categories
            self.categories = {}
            
            # Load categories and entries
            for category_name, entries in data.items():
                category = self.add_category(category_name)
                
                for key, entry_data in entries.items():
                    entry = KnowledgeEntry(
                        key=key,
                        value=entry_data["value"],
                        confidence=entry_data["confidence"],
                        source=entry_data["source"],
                        timestamp=entry_data["timestamp"],
                        times_observed=entry_data["times_observed"],
                        related_keys=entry_data["related_keys"]
                    )
                    category.add_entry(entry)
            
            logger.info(f"Knowledge base loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return False


class KnowledgeExpansionManager:
    """Manager for knowledge expansion and discovery"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge expansion manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.knowledge_base = KnowledgeBase()
        self.observation_queue: List[Dict[str, Any]] = []
        self.max_queue_size = config.get("learning", {}).get("knowledge_queue_size", 1000)
        self.min_confidence_threshold = config.get("learning", {}).get("min_confidence", 0.3)
        self.inference_confidence = config.get("learning", {}).get("inference_confidence", 0.6)
        
        # Pattern detectors for different types of knowledge
        self.pattern_detectors = {
            "npc_detection": self._detect_npc_patterns,
            "quest_detection": self._detect_quest_patterns,
            "location_detection": self._detect_location_patterns,
            "item_detection": self._detect_item_patterns,
            "mechanic_detection": self._detect_mechanic_patterns
        }
        
        # Load existing knowledge base if available
        self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> None:
        """Load knowledge base from disk"""
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "game_knowledge"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(knowledge_dir, exist_ok=True)
        
        # Path for saved knowledge base
        kb_path = os.path.join(knowledge_dir, "learned_knowledge.pkl")
        
        # Try to load knowledge base
        if os.path.exists(kb_path):
            if self.knowledge_base.load(kb_path):
                logger.info("Loaded knowledge base from disk")
    
    def save_knowledge_base(self) -> None:
        """Save knowledge base to disk"""
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "game_knowledge"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(knowledge_dir, exist_ok=True)
        
        # Path for saved knowledge base
        kb_path = os.path.join(knowledge_dir, "learned_knowledge.pkl")
        
        # Save knowledge base
        self.knowledge_base.save(kb_path)
    
    def add_observation(self, observation: Dict[str, Any]) -> None:
        """
        Add a new observation to the queue for processing
        
        Args:
            observation: The observation data
        """
        self.observation_queue.append(observation)
        
        # Trim queue if it gets too large
        if len(self.observation_queue) > self.max_queue_size:
            self.observation_queue = self.observation_queue[-self.max_queue_size:]
    
    def process_observations(self, batch_size: int = 10) -> None:
        """
        Process a batch of observations to extract knowledge
        
        Args:
            batch_size: Number of observations to process
        """
        if not self.observation_queue:
            return
        
        # Process up to batch_size observations
        process_count = min(batch_size, len(self.observation_queue))
        observations = self.observation_queue[:process_count]
        self.observation_queue = self.observation_queue[process_count:]
        
        for observation in observations:
            # Apply pattern detectors to extract knowledge
            for detector_name, detector_func in self.pattern_detectors.items():
                detector_func(observation)
        
        # After processing observations, look for potential inferences
        self._make_inferences()
    
    def _detect_npc_patterns(self, observation: Dict[str, Any]) -> None:
        """
        Detect patterns related to NPCs
        
        Args:
            observation: The observation data
        """
        # Check if observation contains NPC data
        if "entities" in observation:
            for entity in observation["entities"]:
                if entity.get("type") == "npc":
                    # Extract NPC information
                    npc_name = entity.get("name")
                    if npc_name:
                        # Create knowledge entry
                        entry = KnowledgeEntry(
                            key=npc_name,
                            value={
                                "location": entity.get("location", "unknown"),
                                "level": entity.get("level", 0),
                                "hostile": entity.get("hostile", False),
                                "type": entity.get("entity_type", "unknown")
                            },
                            confidence=0.7,  # Direct observation has high confidence
                            source="observation"
                        )
                        
                        # Add to knowledge base
                        self.knowledge_base.add_entry("npcs", entry)
    
    def _detect_quest_patterns(self, observation: Dict[str, Any]) -> None:
        """
        Detect patterns related to quests
        
        Args:
            observation: The observation data
        """
        # Check if observation contains quest data
        if "quest_log" in observation:
            for quest in observation["quest_log"]:
                quest_name = quest.get("name")
                if quest_name:
                    # Create knowledge entry
                    entry = KnowledgeEntry(
                        key=quest_name,
                        value={
                            "description": quest.get("description", ""),
                            "objectives": quest.get("objectives", []),
                            "level": quest.get("level", 0),
                            "rewards": quest.get("rewards", []),
                            "status": quest.get("status", "active")
                        },
                        confidence=0.8,  # Quest log entries are highly reliable
                        source="observation"
                    )
                    
                    # Add to knowledge base
                    self.knowledge_base.add_entry("quests", entry)
    
    def _detect_location_patterns(self, observation: Dict[str, Any]) -> None:
        """
        Detect patterns related to locations
        
        Args:
            observation: The observation data
        """
        # Check if observation contains location data
        if "player_location" in observation:
            location_name = observation["player_location"].get("zone")
            if location_name:
                # Create knowledge entry
                entry = KnowledgeEntry(
                    key=location_name,
                    value={
                        "subzone": observation["player_location"].get("subzone", ""),
                        "coordinates": observation["player_location"].get("coordinates", (0, 0)),
                        "terrain": observation["player_location"].get("terrain", "unknown"),
                        "connected_zones": observation["player_location"].get("connected_zones", [])
                    },
                    confidence=0.9,  # Current location is highly reliable
                    source="observation"
                )
                
                # Add to knowledge base
                self.knowledge_base.add_entry("locations", entry)
    
    def _detect_item_patterns(self, observation: Dict[str, Any]) -> None:
        """
        Detect patterns related to items
        
        Args:
            observation: The observation data
        """
        # Check if observation contains inventory data
        if "inventory" in observation:
            for item in observation["inventory"]:
                item_name = item.get("name")
                if item_name:
                    # Create knowledge entry
                    entry = KnowledgeEntry(
                        key=item_name,
                        value={
                            "type": item.get("type", "unknown"),
                            "quality": item.get("quality", "common"),
                            "level": item.get("level", 0),
                            "stats": item.get("stats", {}),
                            "description": item.get("description", ""),
                            "sell_value": item.get("sell_value", 0)
                        },
                        confidence=0.85,  # Inventory items are reliable
                        source="observation"
                    )
                    
                    # Add to knowledge base
                    self.knowledge_base.add_entry("items", entry)
    
    def _detect_mechanic_patterns(self, observation: Dict[str, Any]) -> None:
        """
        Detect patterns related to game mechanics
        
        Args:
            observation: The observation data
        """
        # Check if observation contains combat data that might reveal mechanics
        if "combat" in observation:
            # Look for spell effects, resistances, etc.
            for spell in observation["combat"].get("spells_used", []):
                spell_name = spell.get("name")
                if spell_name:
                    # Create knowledge entry
                    entry = KnowledgeEntry(
                        key=spell_name,
                        value={
                            "effect": spell.get("effect", "unknown"),
                            "damage_type": spell.get("damage_type", "unknown"),
                            "cast_time": spell.get("cast_time", 0),
                            "cooldown": spell.get("cooldown", 0),
                            "resource_cost": spell.get("resource_cost", 0)
                        },
                        confidence=0.7,  # Combat observations might be situational
                        source="observation"
                    )
                    
                    # Add to knowledge base
                    self.knowledge_base.add_entry("game_mechanics", entry)
    
    def _make_inferences(self) -> None:
        """Make inferences based on existing knowledge"""
        # Example: Infer dungeon boss locations from quest objectives
        quest_entries = self.knowledge_base.categories.get("quests", KnowledgeCategory("quests")).get_all_entries()
        
        for quest_entry in quest_entries:
            quest_data = quest_entry.value
            objectives = quest_data.get("objectives", [])
            
            for objective in objectives:
                if isinstance(objective, str) and "defeat" in objective.lower():
                    # Try to extract boss name
                    parts = objective.lower().split("defeat ")
                    if len(parts) > 1:
                        potential_boss = parts[1].split(" in ")[0].strip()
                        
                        # Check if there's mention of a dungeon
                        if " in " in objective.lower():
                            potential_dungeon = objective.lower().split(" in ")[1].strip()
                            
                            # Create knowledge entries for boss and location
                            boss_entry = KnowledgeEntry(
                                key=potential_boss,
                                value={
                                    "type": "boss",
                                    "location": potential_dungeon,
                                    "related_quest": quest_entry.key
                                },
                                confidence=self.inference_confidence,
                                source="inference",
                                related_keys=[quest_entry.key, potential_dungeon]
                            )
                            
                            # Add to knowledge base
                            self.knowledge_base.add_entry("npcs", boss_entry)
                            
                            # Link back from location to boss
                            location_entry = self.knowledge_base.get_entry("locations", potential_dungeon)
                            if location_entry:
                                if "bosses" not in location_entry.value:
                                    location_entry.value["bosses"] = []
                                if potential_boss not in location_entry.value["bosses"]:
                                    location_entry.value["bosses"].append(potential_boss)
    
    def validate_knowledge(self) -> None:
        """Validate knowledge entries and update confidence"""
        # Review knowledge entries with low confidence
        for category_name, category in self.knowledge_base.categories.items():
            low_confidence_entries = [
                entry for entry in category.get_all_entries() 
                if entry.confidence < self.min_confidence_threshold
            ]
            
            # Decay confidence of old, unverified observations
            current_time = time.time()
            for entry in low_confidence_entries:
                # If entry is old (>7 days) and only observed once, reduce confidence
                if current_time - entry.timestamp > 7 * 24 * 60 * 60 and entry.times_observed == 1:
                    entry.confidence *= 0.9  # Decay confidence by 10%
    
    def get_knowledge(self, category: str, key: str) -> Optional[Any]:
        """
        Get a specific piece of knowledge
        
        Args:
            category: Knowledge category
            key: Knowledge key
            
        Returns:
            The knowledge value or None if not found or confidence too low
        """
        entry = self.knowledge_base.get_entry(category, key)
        
        if entry and entry.confidence >= self.min_confidence_threshold:
            return entry.value
        return None
    
    def get_high_confidence_knowledge(self, category: str) -> Dict[str, Any]:
        """
        Get all high-confidence knowledge for a category
        
        Args:
            category: The category to get knowledge from
            
        Returns:
            Dictionary of high-confidence knowledge
        """
        if category not in self.knowledge_base.categories:
            return {}
        
        return {
            entry.key: entry.value 
            for entry in self.knowledge_base.categories[category].get_entries_by_confidence(self.min_confidence_threshold)
        }
    
    def export_knowledge_to_json(self, output_path: str) -> None:
        """
        Export knowledge base to a JSON file
        
        Args:
            output_path: Path to save the JSON file
        """
        data = {
            category_name: {
                entry.key: {
                    "value": entry.value,
                    "confidence": entry.confidence,
                    "source": entry.source,
                    "timestamp": entry.timestamp,
                    "times_observed": entry.times_observed
                }
                for entry in category.get_entries_by_confidence(self.min_confidence_threshold)
            }
            for category_name, category in self.knowledge_base.categories.items()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Knowledge base exported to {output_path}")
    
    def import_knowledge_from_json(self, input_path: str) -> bool:
        """
        Import knowledge base from a JSON file
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            Success status
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
                
            for category_name, entries in data.items():
                for key, entry_data in entries.items():
                    entry = KnowledgeEntry(
                        key=key,
                        value=entry_data["value"],
                        confidence=entry_data["confidence"],
                        source=entry_data["source"],
                        timestamp=entry_data["timestamp"],
                        times_observed=entry_data["times_observed"],
                        related_keys=[]
                    )
                    self.knowledge_base.add_entry(category_name, entry)
            
            logger.info(f"Knowledge imported from {input_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to import knowledge: {e}")
            return False