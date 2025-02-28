"""
Combat Testing Module

This module provides testing capabilities for the combat system, analyzing
the effectiveness of various combat modules and configurations.
"""

import logging
import time
import random
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.decision.combat.base_combat_module import BaseCombatModule
from src.decision.combat_manager import CombatManager
from src.decision.combat.situational_awareness import CombatSituationalAwareness


class CombatTester:
    """
    Combat tester for evaluating and analyzing combat performance
    
    This class can perform automated tests on different combat modules
    to assess their effectiveness, optimize rotations, and identify
    edge cases. It provides detailed performance metrics and comparisons.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge: GameKnowledge):
        """
        Initialize the combat tester
        
        Args:
            config: Configuration dictionary
            knowledge: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.decision.combat_testing")
        self.config = config
        self.knowledge = knowledge
        
        # Set up the combat manager to test
        self.combat_manager = CombatManager(config, knowledge)
        self.situational_awareness = CombatSituationalAwareness(config, knowledge)
        
        # Test results storage
        self.test_results: Dict[str, Any] = {}
        self.current_test_name: Optional[str] = None
        
        # Test scenario parameters
        self.test_duration: float = 60.0  # Default 60-second test
        self.test_scenario: Dict[str, Any] = {}
        
        # Test metrics
        self.dps_results: Dict[str, float] = {}
        self.rotation_results: Dict[str, List[Dict[str, Any]]] = {}
        self.resource_usage: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("Combat tester initialized")
    
    def setup_test(self, test_name: str, scenario: Dict[str, Any], duration: float = 60.0) -> None:
        """
        Set up a combat test scenario
        
        Args:
            test_name: Name of the test
            scenario: Test scenario parameters including enemy types, player class, etc.
            duration: Duration of test in seconds
        """
        self.current_test_name = test_name
        self.test_scenario = scenario
        self.test_duration = duration
        
        # Reset test metrics
        self.test_results = {
            "name": test_name,
            "scenario": scenario,
            "duration": duration,
            "start_time": time.time(),
            "end_time": None,
            "dps": 0.0,
            "damage_breakdown": {},
            "resource_usage": {},
            "rotation_analysis": [],
            "situational_responses": [],
            "combat_logs": []
        }
        
        self.logger.info(f"Set up test: {test_name} ({duration}s)")
    
    def run_test(self) -> Dict[str, Any]:
        """
        Run the configured combat test
        
        Returns:
            Dict: Test results
        """
        if not self.current_test_name or not self.test_scenario:
            self.logger.error("Cannot run test: no test configured")
            return {}
        
        self.logger.info(f"Starting test: {self.current_test_name}")
        
        # Set up initial state based on scenario
        initial_state = self._create_test_state()
        
        # Run simulated combat for the test duration
        start_time = time.time()
        end_time = start_time + self.test_duration
        current_time = start_time
        
        total_damage = 0.0
        ability_damage = {}
        resource_state = {}
        
        # Capture each step in the rotation for analysis
        rotation_sequence = []
        situational_responses = []
        
        # Main simulation loop
        while current_time < end_time:
            # Update state with current time and other dynamic factors
            state = self._update_test_state(initial_state, current_time - start_time)
            
            # Update situational awareness
            self.situational_awareness.update(state)
            tactical_suggestions = self.situational_awareness.get_tactical_suggestions()
            
            # Record situational awareness data
            if tactical_suggestions:
                situational_responses.append({
                    "time": current_time - start_time,
                    "suggestions": tactical_suggestions,
                    "awareness": {
                        "aoe_opportunity": self.situational_awareness.aoe_opportunity,
                        "interrupt_targets": len(self.situational_awareness.interrupt_targets),
                        "danger_level": self.situational_awareness.danger_level
                    }
                })
            
            # Generate combat plan
            combat_plan = self.combat_manager.generate_combat_plan(state)
            
            # Execute next ability in plan
            if combat_plan:
                next_action = combat_plan[0]
                
                # Calculate damage and resource changes
                damage, resources_used = self._simulate_ability_execution(
                    next_action, state, current_time - start_time)
                
                # Record results
                total_damage += damage
                ability_name = next_action.get("spell", "Unknown")
                ability_damage[ability_name] = ability_damage.get(ability_name, 0) + damage
                
                # Update resource tracking
                for resource, amount in resources_used.items():
                    resource_state[resource] = resource_state.get(resource, 0) + amount
                
                # Record this step in the rotation
                rotation_sequence.append({
                    "time": current_time - start_time,
                    "action": next_action,
                    "damage": damage,
                    "resources": resources_used,
                    "resource_state": resource_state.copy()
                })
                
                # Record combat log
                self.test_results["combat_logs"].append({
                    "time": current_time - start_time,
                    "action": next_action,
                    "damage": damage,
                    "resources": resources_used
                })
            
            # Advance time
            current_time += self._get_action_time(state)
        
        # Calculate final results
        actual_duration = time.time() - start_time
        dps = total_damage / self.test_duration
        
        # Compile test results
        self.test_results["end_time"] = time.time()
        self.test_results["actual_duration"] = actual_duration
        self.test_results["dps"] = dps
        self.test_results["damage_breakdown"] = ability_damage
        self.test_results["resource_usage"] = resource_state
        self.test_results["rotation_analysis"] = rotation_sequence
        self.test_results["situational_responses"] = situational_responses
        
        # Log key results
        self.logger.info(f"Test completed: {self.current_test_name}")
        self.logger.info(f"DPS: {dps:.2f}")
        self.logger.info(f"Top abilities: {sorted(ability_damage.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        return self.test_results
    
    def _create_test_state(self) -> GameState:
        """
        Create initial game state for the test based on test scenario
        
        Returns:
            GameState: Initial game state
        """
        state = GameState()
        
        # Set player class and level
        state.player_class = self.test_scenario.get("player_class", "warrior")
        state.player_level = self.test_scenario.get("player_level", 60)
        
        # Set resources
        state.player_health = self.test_scenario.get("player_health", 100)
        state.player_health_max = self.test_scenario.get("player_health_max", 100)
        state.player_mana = self.test_scenario.get("player_mana", 100)
        state.player_mana_max = self.test_scenario.get("player_mana_max", 100)
        state.player_rage = self.test_scenario.get("player_rage", 0)
        state.player_rage_max = self.test_scenario.get("player_rage_max", 100)
        state.player_energy = self.test_scenario.get("player_energy", 100)
        state.player_energy_max = self.test_scenario.get("player_energy_max", 100)
        state.player_focus = self.test_scenario.get("player_focus", 100)
        state.player_focus_max = self.test_scenario.get("player_focus_max", 100)
        
        # Set target
        state.target = self.test_scenario.get("target_id", "Target Dummy")
        state.target_health = self.test_scenario.get("target_health", 100)
        state.target_level = self.test_scenario.get("target_level", 60)
        
        # Set combat status
        state.is_in_combat = self.test_scenario.get("in_combat", True)
        
        # Set up nearby entities
        state.nearby_entities = self.test_scenario.get("nearby_entities", [])
        if not state.nearby_entities and state.target:
            # Add at least the target as a nearby entity
            state.nearby_entities = [{
                "id": state.target,
                "type": "mob",
                "reaction": "hostile",
                "health_percent": state.target_health,
                "level": state.target_level,
                "position": (10, 0),  # 10 yards away by default
                "casting": None
            }]
        
        # Set up player position
        state.player_position = (0, 0)
        
        # Set up player buffs
        state.player_buffs = self.test_scenario.get("player_buffs", [])
        
        # Set up target debuffs
        state.target_debuffs = self.test_scenario.get("target_debuffs", {})
        
        return state
    
    def _update_test_state(self, base_state: GameState, elapsed_time: float) -> GameState:
        """
        Update game state based on elapsed time and previous actions
        
        Args:
            base_state: Initial game state
            elapsed_time: Time elapsed since test start
            
        Returns:
            GameState: Updated game state
        """
        state = GameState()
        
        # Copy base state
        for attr in dir(base_state):
            if not attr.startswith("__") and not callable(getattr(base_state, attr)):
                setattr(state, attr, getattr(base_state, attr))
        
        # Update resources based on regeneration or depletion
        if hasattr(state, "player_mana"):
            # Simulate mana regeneration (% per second)
            regen_rate = self.test_scenario.get("mana_regen_rate", 1.0)
            max_mana = getattr(state, "player_mana_max", 100)
            new_mana = min(getattr(state, "player_mana") + (regen_rate * max_mana / 100), max_mana)
            setattr(state, "player_mana", new_mana)
        
        if hasattr(state, "player_energy"):
            # Simulate energy regeneration (flat amount per second)
            regen_rate = self.test_scenario.get("energy_regen_rate", 10.0)
            max_energy = getattr(state, "player_energy_max", 100)
            new_energy = min(getattr(state, "player_energy") + regen_rate, max_energy)
            setattr(state, "player_energy", new_energy)
        
        if hasattr(state, "player_focus"):
            # Simulate focus regeneration
            regen_rate = self.test_scenario.get("focus_regen_rate", 5.0)
            max_focus = getattr(state, "player_focus_max", 100)
            new_focus = min(getattr(state, "player_focus") + regen_rate, max_focus)
            setattr(state, "player_focus", new_focus)
        
        # Simulate rage decay
        if hasattr(state, "player_rage"):
            decay_rate = self.test_scenario.get("rage_decay_rate", 0.5)
            new_rage = max(getattr(state, "player_rage") - decay_rate, 0)
            setattr(state, "player_rage", new_rage)
        
        # Update debuff durations on target
        updated_debuffs = {}
        for debuff, data in getattr(state, "target_debuffs", {}).items():
            if data.get("duration", 0) > elapsed_time - data.get("applied_at", 0):
                updated_debuffs[debuff] = data
        state.target_debuffs = updated_debuffs
        
        # Update buff durations on player
        updated_buffs = []
        for buff in getattr(state, "player_buffs", []):
            # For simplicity, assume all buffs last for test duration
            # In a more complex simulation, would track individual buff durations
            updated_buffs.append(buff)
        state.player_buffs = updated_buffs
        
        # Update enemy casting state (for interrupt testing)
        if self.test_scenario.get("test_interrupts", False):
            for entity in getattr(state, "nearby_entities", []):
                # Randomly start casting on some enemies
                if entity.get("type") == "mob" and random.random() < 0.1:
                    spell_name = random.choice(["Fireball", "Shadow Bolt", "Heal", "Frostbolt"])
                    entity["casting"] = {
                        "spell": spell_name,
                        "remaining_time": 2.0
                    }
        
        # Add dynamic combat events for situational testing
        if self.test_scenario.get("test_situational_awareness", False):
            # Occasionally add dangerous abilities
            if random.random() < 0.05:
                state.ground_effects = [{
                    "name": "Fire Zone",
                    "position": (random.uniform(-5, 5), random.uniform(-5, 5)),
                    "radius": 3.0,
                    "remaining_time": 5.0
                }]
            
            # Occasionally change enemy count for AoE testing
            if random.random() < 0.1:
                new_enemy_count = random.randint(1, 5)
                state.nearby_entities = []
                for i in range(new_enemy_count):
                    state.nearby_entities.append({
                        "id": f"Enemy_{i}",
                        "type": "mob",
                        "reaction": "hostile",
                        "health_percent": 100,
                        "level": state.player_level,
                        "position": (random.uniform(5, 15), random.uniform(-5, 5)),
                        "casting": None
                    })
        
        return state
    
    def _simulate_ability_execution(self, action: Dict[str, Any], state: GameState, 
                                   time_point: float) -> Tuple[float, Dict[str, float]]:
        """
        Simulate execution of an ability to determine damage and resource changes
        
        Args:
            action: Combat action to execute
            state: Current game state
            time_point: Time point in the simulation
            
        Returns:
            Tuple[float, Dict[str, float]]: Damage done and resources used
        """
        ability_name = action.get("spell", "Unknown")
        resources_used = {}
        
        # Get base damage from knowledge base or scenario data
        ability_info = self.knowledge.get_ability_info(ability_name, state.player_class)
        
        # Default damage values if not available in knowledge base
        base_damage = ability_info.get("base_damage", 0) if ability_info else 0
        
        # Apply modifiers based on buffs and debuffs
        damage_modifier = 1.0
        
        # Check for buffs that increase damage
        for buff in getattr(state, "player_buffs", []):
            buff_info = self.knowledge.get_buff_info(buff)
            if buff_info and "damage_modifier" in buff_info:
                damage_modifier *= (1 + buff_info["damage_modifier"] / 100)
        
        # Check for debuffs that increase damage taken
        for debuff, data in getattr(state, "target_debuffs", {}).items():
            debuff_info = self.knowledge.get_debuff_info(debuff)
            if debuff_info and "damage_taken_modifier" in debuff_info:
                damage_modifier *= (1 + debuff_info["damage_taken_modifier"] / 100)
        
        # Calculate final damage
        # Add variance to simulate real combat conditions
        variance = random.uniform(0.9, 1.1)
        final_damage = base_damage * damage_modifier * variance
        
        # Check for critical hits
        crit_chance = self.test_scenario.get("crit_chance", 20)
        if random.randint(1, 100) <= crit_chance:
            final_damage *= 2.0
        
        # Calculate resource costs
        if state.player_class == "warrior":
            # Rage-based
            rage_cost = ability_info.get("rage_cost", 0) if ability_info else 0
            if rage_cost > 0:
                resources_used["rage"] = rage_cost
        elif state.player_class in ["mage", "priest", "warlock"]:
            # Mana-based
            mana_cost = ability_info.get("mana_cost", 0) if ability_info else 0
            if mana_cost > 0:
                resources_used["mana"] = mana_cost
        elif state.player_class == "rogue":
            # Energy-based
            energy_cost = ability_info.get("energy_cost", 0) if ability_info else 0
            if energy_cost > 0:
                resources_used["energy"] = energy_cost
        elif state.player_class == "hunter":
            # Focus-based
            focus_cost = ability_info.get("focus_cost", 0) if ability_info else 0
            if focus_cost != 0:  # Can be negative for focus generators
                resources_used["focus"] = focus_cost
        
        # Record debuff application
        if ability_info and ability_info.get("applies_debuff"):
            debuff_name = ability_info["applies_debuff"]
            debuff_duration = ability_info.get("debuff_duration", 15)
            if not hasattr(state, "target_debuffs"):
                state.target_debuffs = {}
            state.target_debuffs[debuff_name] = {
                "applied_at": time_point,
                "duration": debuff_duration,
                "source": state.player_class
            }
        
        # Record buff application
        if ability_info and ability_info.get("applies_buff"):
            buff_name = ability_info["applies_buff"]
            if not hasattr(state, "player_buffs"):
                state.player_buffs = []
            if buff_name not in state.player_buffs:
                state.player_buffs.append(buff_name)
        
        return final_damage, resources_used
    
    def _get_action_time(self, state: GameState) -> float:
        """
        Calculate the time each action takes
        
        Args:
            state: Current game state
            
        Returns:
            float: Time in seconds for the action
        """
        global_cooldown = 1.5  # Standard GCD
        
        # Factor in haste from buffs
        haste_modifier = 1.0
        for buff in getattr(state, "player_buffs", []):
            buff_info = self.knowledge.get_buff_info(buff)
            if buff_info and "haste_modifier" in buff_info:
                haste_modifier *= (1 - buff_info["haste_modifier"] / 100)
        
        # Apply haste and floor at 0.75 seconds (minimum GCD)
        adjusted_gcd = max(global_cooldown * haste_modifier, 0.75)
        
        return adjusted_gcd
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save test results to a JSON file
        
        Args:
            filename: Optional filename to save results to
            
        Returns:
            str: Path to saved results file
        """
        if not self.test_results:
            self.logger.error("No test results to save")
            return ""
        
        if not filename:
            # Generate filename based on test name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_name = self.current_test_name.replace(" ", "_").lower() if self.current_test_name else "unknown_test"
            filename = f"combat_test_{test_name}_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs("data/test_results", exist_ok=True)
        filepath = os.path.join("data/test_results", filename)
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        self.logger.info(f"Test results saved to {filepath}")
        return filepath
    
    def compare_tests(self, test_result1: Dict[str, Any], test_result2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two test results to identify differences
        
        Args:
            test_result1: First test result
            test_result2: Second test result
            
        Returns:
            Dict: Comparison results
        """
        # Basic validation
        if not test_result1 or not test_result2:
            self.logger.error("Cannot compare: missing test results")
            return {"error": "Missing test results"}
        
        comparison = {
            "test1_name": test_result1.get("name", "Unknown Test 1"),
            "test2_name": test_result2.get("name", "Unknown Test 2"),
            "dps_difference": test_result2.get("dps", 0) - test_result1.get("dps", 0),
            "dps_percent_change": ((test_result2.get("dps", 0) / max(test_result1.get("dps", 1), 1)) - 1) * 100,
            "damage_comparison": {},
            "resource_comparison": {},
            "rotation_differences": []
        }
        
        # Compare damage breakdown
        damage1 = test_result1.get("damage_breakdown", {})
        damage2 = test_result2.get("damage_breakdown", {})
        
        all_abilities = set(damage1.keys()) | set(damage2.keys())
        for ability in all_abilities:
            value1 = damage1.get(ability, 0)
            value2 = damage2.get(ability, 0)
            comparison["damage_comparison"][ability] = {
                "test1": value1,
                "test2": value2,
                "difference": value2 - value1,
                "percent_change": ((value2 / max(value1, 1)) - 1) * 100 if value1 > 0 else float('inf')
            }
        
        # Compare resource usage
        resource1 = test_result1.get("resource_usage", {})
        resource2 = test_result2.get("resource_usage", {})
        
        all_resources = set(resource1.keys()) | set(resource2.keys())
        for resource in all_resources:
            value1 = resource1.get(resource, 0)
            value2 = resource2.get(resource, 0)
            comparison["resource_comparison"][resource] = {
                "test1": value1,
                "test2": value2,
                "difference": value2 - value1,
                "percent_change": ((value2 / max(value1, 1)) - 1) * 100 if value1 > 0 else float('inf')
            }
        
        # Compare rotation differences
        rotation1 = test_result1.get("rotation_analysis", [])
        rotation2 = test_result2.get("rotation_analysis", [])
        
        # Analyze which abilities were used more or less frequently
        ability_count1 = {}
        ability_count2 = {}
        
        for action in rotation1:
            ability = action.get("action", {}).get("spell", "Unknown")
            ability_count1[ability] = ability_count1.get(ability, 0) + 1
        
        for action in rotation2:
            ability = action.get("action", {}).get("spell", "Unknown")
            ability_count2[ability] = ability_count2.get(ability, 0) + 1
        
        all_rotation_abilities = set(ability_count1.keys()) | set(ability_count2.keys())
        for ability in all_rotation_abilities:
            count1 = ability_count1.get(ability, 0)
            count2 = ability_count2.get(ability, 0)
            comparison["rotation_differences"].append({
                "ability": ability,
                "test1_count": count1,
                "test2_count": count2,
                "difference": count2 - count1
            })
        
        # Sort rotation differences by absolute difference
        comparison["rotation_differences"].sort(
            key=lambda x: abs(x["difference"]), reverse=True)
        
        return comparison


def run_standard_tests() -> None:
    """
    Run a standard suite of combat tests for different classes and scenarios
    """
    logger = logging.getLogger("wow_ai.decision.combat_testing.standard_tests")
    logger.info("Starting standard combat tests")
    
    # Load config and knowledge
    config = {}
    knowledge = GameKnowledge({})
    
    # Initialize tester
    tester = CombatTester(config, knowledge)
    
    # Test each class
    class_scenarios = [
        {
            "name": "Warrior (Arms)",
            "player_class": "warrior",
            "warrior_spec": "arms",
            "player_level": 60
        },
        {
            "name": "Warrior (Fury)",
            "player_class": "warrior",
            "warrior_spec": "fury",
            "player_level": 60
        },
        {
            "name": "Mage (Frost)",
            "player_class": "mage",
            "mage_spec": "frost",
            "player_level": 60
        },
        {
            "name": "Mage (Fire)",
            "player_class": "mage",
            "mage_spec": "fire",
            "player_level": 60
        },
        {
            "name": "Priest (Shadow)",
            "player_class": "priest",
            "priest_spec": "shadow",
            "player_level": 60,
            "player_buffs": ["Shadowform"]
        },
        {
            "name": "Hunter (Beast Mastery)",
            "player_class": "hunter",
            "hunter_spec": "beast_mastery",
            "player_level": 60,
            "pet_active": True
        }
    ]
    
    results = []
    
    for scenario in class_scenarios:
        test_name = f"Standard Test - {scenario['name']}"
        tester.setup_test(test_name, scenario, duration=120.0)
        
        logger.info(f"Running test: {test_name}")
        test_result = tester.run_test()
        results.append(test_result)
        
        # Save test results
        tester.save_results()
    
    # Run situational awareness test
    situational_test = {
        "name": "Warrior (Arms) - Situational",
        "player_class": "warrior",
        "warrior_spec": "arms",
        "player_level": 60,
        "test_situational_awareness": True,
        "test_interrupts": True
    }
    
    test_name = f"Situational Test - {situational_test['name']}"
    tester.setup_test(test_name, situational_test, duration=120.0)
    
    logger.info(f"Running test: {test_name}")
    situational_result = tester.run_test()
    results.append(situational_result)
    
    # Save test results
    tester.save_results()
    
    # Compare best results
    if len(results) >= 2:
        best_result = max(results, key=lambda x: x.get("dps", 0))
        worst_result = min(results, key=lambda x: x.get("dps", 0))
        
        logger.info(f"Best performing test: {best_result.get('name')} with DPS: {best_result.get('dps', 0):.2f}")
        logger.info(f"Worst performing test: {worst_result.get('name')} with DPS: {worst_result.get('dps', 0):.2f}")
        
        comparison = tester.compare_tests(worst_result, best_result)
        logger.info(f"DPS difference: {comparison['dps_difference']:.2f} ({comparison['dps_percent_change']:.2f}%)")
    
    logger.info("Standard combat tests completed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run standard tests
    run_standard_tests()