"""
Transfer Learning System

This module implements mechanisms for transferring knowledge and skills
between different tasks, character classes, or game situations to enable
faster learning and adaptation.
"""

import logging
import os
import pickle
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

# Setup module-level logger
logger = logging.getLogger("wow_ai.learning.transfer_learning")

@dataclass
class TransferableSkill:
    """Represents a skill that can be transferred between contexts"""
    name: str
    description: str
    source_context: str  # e.g., "warrior_combat", "goldshire_questing"
    parameters: Dict[str, Any] = field(default_factory=dict)
    transfer_count: int = 0
    success_rate: float = 0.0
    applicable_contexts: List[str] = field(default_factory=list)


class SkillTransferRules:
    """Rules for determining skill transferability between contexts"""
    
    def __init__(self):
        """Initialize transfer rules"""
        self.similarity_thresholds: Dict[str, float] = {
            "class_to_class": 0.6,  # Threshold for transferring between character classes
            "zone_to_zone": 0.8,    # Threshold for transferring between zones
            "combat_to_combat": 0.7, # Threshold for transferring between combat situations
            "quest_to_quest": 0.9    # Threshold for transferring between quest types
        }
        
        # Context similarity map (pre-defined similarities)
        # Higher value = more similar contexts
        self.context_similarity: Dict[Tuple[str, str], float] = {
            # Class similarities
            ("warrior", "paladin"): 0.7,
            ("warrior", "rogue"): 0.5,
            ("warrior", "hunter"): 0.4,
            ("mage", "warlock"): 0.6,
            ("mage", "priest"): 0.5,
            ("druid", "shaman"): 0.6,
            
            # Zone similarities
            ("elwynn_forest", "westfall"): 0.8,
            ("dun_morogh", "loch_modan"): 0.8,
            ("durotar", "barrens"): 0.8,
            ("teldrassil", "darkshore"): 0.8,
            
            # Combat type similarities
            ("melee_single_target", "melee_multi_target"): 0.7,
            ("ranged_single_target", "ranged_multi_target"): 0.7,
            ("tank_dungeon", "tank_raid"): 0.6
        }
    
    def get_similarity(self, context1: str, context2: str) -> float:
        """
        Get similarity score between two contexts
        
        Args:
            context1: First context
            context2: Second context
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # If contexts are the same, return perfect similarity
        if context1 == context2:
            return 1.0
        
        # Check if we have a pre-defined similarity
        if (context1, context2) in self.context_similarity:
            return self.context_similarity[(context1, context2)]
        
        # Check the reverse order
        if (context2, context1) in self.context_similarity:
            return self.context_similarity[(context2, context1)]
        
        # If we don't have a pre-defined similarity, use a heuristic
        # based on the context names
        parts1 = context1.split('_')
        parts2 = context2.split('_')
        
        # Count matching parts
        matches = sum(1 for p1 in parts1 if any(p1 == p2 for p2 in parts2))
        
        # Calculate similarity based on matching parts
        similarity = matches / max(len(parts1), len(parts2))
        
        return similarity
    
    def is_transferable(self, source_context: str, target_context: str, threshold_key: str = None) -> bool:
        """
        Determine if a skill is transferable between contexts
        
        Args:
            source_context: Context where skill was learned
            target_context: Context where skill might be applied
            threshold_key: Key for specific threshold, or None for default
            
        Returns:
            True if skill is transferable, False otherwise
        """
        similarity = self.get_similarity(source_context, target_context)
        
        if threshold_key and threshold_key in self.similarity_thresholds:
            return similarity >= self.similarity_thresholds[threshold_key]
        
        # Default threshold if no specific key provided
        return similarity >= 0.5


class TransferLearningManager:
    """Manager for transfer learning capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transfer learning manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.skills: Dict[str, TransferableSkill] = {}
        self.rules = SkillTransferRules()
        self.context_mapping: Dict[str, List[str]] = defaultdict(list)
        
        # Transfer success tracking
        self.transfer_attempts: int = 0
        self.transfer_successes: int = 0
        
        # Load existing skills if available
        self._load_skills()
    
    def _load_skills(self) -> None:
        """Load transferable skills from disk"""
        skills_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(skills_dir, exist_ok=True)
        
        # Path for saved skills
        skills_path = os.path.join(skills_dir, "transferable_skills.pkl")
        
        # Try to load skills
        if os.path.exists(skills_path):
            try:
                with open(skills_path, 'rb') as f:
                    self.skills = pickle.load(f)
                logger.info("Loaded transferable skills from disk")
                
                # Rebuild context mapping
                self._rebuild_context_mapping()
            except Exception as e:
                logger.error(f"Failed to load transferable skills: {e}")
    
    def _rebuild_context_mapping(self) -> None:
        """Rebuild the context mapping from loaded skills"""
        self.context_mapping = defaultdict(list)
        for skill_name, skill in self.skills.items():
            self.context_mapping[skill.source_context].append(skill_name)
            for context in skill.applicable_contexts:
                if context != skill.source_context:
                    self.context_mapping[context].append(skill_name)
    
    def save_skills(self) -> None:
        """Save transferable skills to disk"""
        skills_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(skills_dir, exist_ok=True)
        
        # Path for saved skills
        skills_path = os.path.join(skills_dir, "transferable_skills.pkl")
        
        try:
            with open(skills_path, 'wb') as f:
                pickle.dump(self.skills, f)
            logger.info("Saved transferable skills to disk")
        except Exception as e:
            logger.error(f"Failed to save transferable skills: {e}")
    
    def register_skill(self, skill: TransferableSkill) -> None:
        """
        Register a new transferable skill
        
        Args:
            skill: The transferable skill to register
        """
        if skill.name in self.skills:
            logger.info(f"Updating existing skill: {skill.name}")
        else:
            logger.info(f"Registering new transferable skill: {skill.name}")
        
        self.skills[skill.name] = skill
        
        # Update context mapping
        self.context_mapping[skill.source_context].append(skill.name)
        for context in skill.applicable_contexts:
            if context != skill.source_context:
                self.context_mapping[context].append(skill.name)
    
    def discover_applicable_contexts(self, skill_name: str) -> List[str]:
        """
        Discover contexts where a skill might be applicable
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            List of applicable context names
        """
        if skill_name not in self.skills:
            return []
        
        skill = self.skills[skill_name]
        discovered_contexts = []
        
        # Check all known contexts for potential transfers
        for context in self.context_mapping.keys():
            if context == skill.source_context:
                continue  # Skip the source context
                
            # Determine the appropriate threshold key
            threshold_key = None
            if "combat" in skill.source_context and "combat" in context:
                threshold_key = "combat_to_combat"
            elif "quest" in skill.source_context and "quest" in context:
                threshold_key = "quest_to_quest"
            
            # Check if the skill is transferable to this context
            if self.rules.is_transferable(skill.source_context, context, threshold_key):
                discovered_contexts.append(context)
                
                # Add to skill's applicable contexts if not already there
                if context not in skill.applicable_contexts:
                    skill.applicable_contexts.append(context)
        
        return discovered_contexts
    
    def get_applicable_skills(self, context: str) -> List[TransferableSkill]:
        """
        Get skills applicable to a specific context
        
        Args:
            context: The target context
            
        Returns:
            List of applicable skills
        """
        applicable_skills = []
        
        # Check skills directly mapped to this context
        for skill_name in self.context_mapping.get(context, []):
            applicable_skills.append(self.skills[skill_name])
        
        # Check other skills for transferability
        for skill_name, skill in self.skills.items():
            if skill_name not in self.context_mapping.get(context, []):
                # Determine the appropriate threshold key
                threshold_key = None
                if "combat" in skill.source_context and "combat" in context:
                    threshold_key = "combat_to_combat"
                elif "quest" in skill.source_context and "quest" in context:
                    threshold_key = "quest_to_quest"
                
                if self.rules.is_transferable(skill.source_context, context, threshold_key):
                    applicable_skills.append(skill)
                    
                    # Update the context mapping
                    self.context_mapping[context].append(skill_name)
                    
                    # Update the skill's applicable contexts
                    if context not in skill.applicable_contexts:
                        skill.applicable_contexts.append(context)
        
        return applicable_skills
    
    def apply_skill(self, skill_name: str, target_context: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply a skill to a target context
        
        Args:
            skill_name: Name of the skill to apply
            target_context: Context to apply the skill to
            parameters: Optional parameters for skill application
            
        Returns:
            Result of skill application
        """
        if skill_name not in self.skills:
            return {"success": False, "error": "Skill not found"}
        
        skill = self.skills[skill_name]
        
        # Determine if skill is transferable to this context
        threshold_key = None
        if "combat" in skill.source_context and "combat" in target_context:
            threshold_key = "combat_to_combat"
        elif "quest" in skill.source_context and "quest" in target_context:
            threshold_key = "quest_to_quest"
        
        is_transferable = self.rules.is_transferable(
            skill.source_context, target_context, threshold_key
        )
        
        if not is_transferable and target_context != skill.source_context:
            return {"success": False, "error": "Skill not transferable to this context"}
        
        # Combine skill parameters with provided parameters
        merged_params = skill.parameters.copy()
        if parameters:
            merged_params.update(parameters)
        
        # Simulate skill application (in a real system, this would execute the skill)
        # For demonstration, we'll just return the parameters and skill info
        result = {
            "success": True,
            "skill_name": skill_name,
            "source_context": skill.source_context,
            "target_context": target_context,
            "parameters": merged_params,
            "is_transfer": target_context != skill.source_context
        }
        
        # Update transfer statistics
        if target_context != skill.source_context:
            self.transfer_attempts += 1
            skill.transfer_count += 1
            
            # In a real system, we would determine success based on actual outcome
            # For demonstration, we'll assume 80% success rate for transfers
            success_probability = 0.8
            transfer_success = np.random.random() < success_probability
            
            if transfer_success:
                self.transfer_successes += 1
                
                # Update skill success rate
                skill.success_rate = ((skill.success_rate * (skill.transfer_count - 1)) + 1.0) / skill.transfer_count
            else:
                # Update skill success rate
                skill.success_rate = (skill.success_rate * (skill.transfer_count - 1)) / skill.transfer_count
            
            result["transfer_success"] = transfer_success
        
        # If this is a new applicable context, add it
        if target_context not in skill.applicable_contexts:
            skill.applicable_contexts.append(target_context)
            self.context_mapping[target_context].append(skill_name)
        
        return result
    
    def get_transfer_success_rate(self) -> float:
        """
        Get overall transfer success rate
        
        Returns:
            Success rate as a float (0.0 to 1.0)
        """
        if self.transfer_attempts == 0:
            return 0.0
        
        return self.transfer_successes / self.transfer_attempts
    
    def get_skill_by_name(self, skill_name: str) -> Optional[TransferableSkill]:
        """
        Get a skill by name
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            The skill or None if not found
        """
        return self.skills.get(skill_name)
    
    def get_skills_by_source(self, source_context: str) -> List[TransferableSkill]:
        """
        Get skills originally learned in a specific context
        
        Args:
            source_context: Source context name
            
        Returns:
            List of skills from that context
        """
        return [skill for skill in self.skills.values() if skill.source_context == source_context]
    
    def analyze_transfer_opportunities(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze potential skill transfer opportunities
        
        Returns:
            Dictionary mapping contexts to lists of potential skill transfers
        """
        opportunities = {}
        
        # For each context, find skills that might be transferable but aren't yet
        for target_context in self.context_mapping.keys():
            potential_transfers = []
            
            for skill_name, skill in self.skills.items():
                # Skip if skill is already applied to this context
                if target_context in skill.applicable_contexts:
                    continue
                
                # Determine the appropriate threshold key
                threshold_key = None
                if "combat" in skill.source_context and "combat" in target_context:
                    threshold_key = "combat_to_combat"
                elif "quest" in skill.source_context and "quest" in target_context:
                    threshold_key = "quest_to_quest"
                
                # Check transferability
                similarity = self.rules.get_similarity(skill.source_context, target_context)
                is_transferable = self.rules.is_transferable(
                    skill.source_context, target_context, threshold_key
                )
                
                if is_transferable:
                    potential_transfers.append({
                        "skill_name": skill_name,
                        "source_context": skill.source_context,
                        "similarity": similarity,
                        "success_rate": skill.success_rate,
                        "transfer_count": skill.transfer_count
                    })
            
            if potential_transfers:
                opportunities[target_context] = potential_transfers
        
        return opportunities
    
    def export_skills_to_json(self, output_path: str) -> None:
        """
        Export transferable skills to a JSON file
        
        Args:
            output_path: Path to save the JSON file
        """
        data = {}
        for skill_name, skill in self.skills.items():
            data[skill_name] = {
                "description": skill.description,
                "source_context": skill.source_context,
                "parameters": skill.parameters,
                "transfer_count": skill.transfer_count,
                "success_rate": skill.success_rate,
                "applicable_contexts": skill.applicable_contexts
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported transferable skills to {output_path}")
    
    def import_skills_from_json(self, input_path: str) -> bool:
        """
        Import transferable skills from a JSON file
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            Success status
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            for skill_name, skill_data in data.items():
                skill = TransferableSkill(
                    name=skill_name,
                    description=skill_data["description"],
                    source_context=skill_data["source_context"],
                    parameters=skill_data["parameters"],
                    transfer_count=skill_data["transfer_count"],
                    success_rate=skill_data["success_rate"],
                    applicable_contexts=skill_data["applicable_contexts"]
                )
                self.register_skill(skill)
            
            logger.info(f"Imported transferable skills from {input_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to import transferable skills: {e}")
            return False