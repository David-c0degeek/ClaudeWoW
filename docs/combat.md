# Combat System Documentation

This document provides detailed information about the combat system in ClaudeWoW, including architecture, class-specific implementations, and combat decision making.

## Overview

The combat system in ClaudeWoW is designed to make intelligent combat decisions for World of Warcraft characters. It supports different classes, specializations, and combat situations through a modular, extensible architecture.

Key features include:
- Class-specific combat rotations
- Resource management (mana, rage, energy, focus)
- Target selection and positioning
- AoE detection and handling
- Interrupt priorities
- Defensive ability usage
- Group role coordination

## Architecture

The combat system is built on a layered architecture:

1. **Combat Manager**: Coordinates all combat decisions and dispatches to class-specific modules
2. **Base Combat Module**: Abstract base class defining the interface for all class modules
3. **Class Combat Modules**: Specialized implementations for each character class
4. **Situational Awareness**: Context-aware analysis of combat situations

### Component Relationships

```
┌────────────────┐     ┌─────────────────────┐
│ Combat Manager │────▶│ Situational         │
└────────────────┘     │ Awareness           │
        │              └─────────────────────┘
        │
        ▼
┌────────────────┐
│ Base Combat    │
│ Module         │
└────────────────┘
        ▲
        │
        ├─────────────┬─────────────┬─────────────┐
        │             │             │             │
┌───────┴─────┐ ┌─────┴───────┐ ┌───┴─────────┐ ┌─┴───────────┐
│ Warrior     │ │ Mage        │ │ Priest      │ │ Other Class │
│ Module      │ │ Module      │ │ Module      │ │ Modules     │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

## Combat Manager

The `CombatManager` class serves as the primary interface for the rest of the system:

1. Loads the appropriate class module based on the player's class
2. Generates combat plans based on the current game state
3. Handles fallback logic when no class module is available
4. Provides interface for querying supported classes and builds

### Key Methods

- `generate_combat_plan(state)`: Creates a sequence of combat actions
- `get_supported_classes()`: Lists available class implementations
- `get_talent_builds_for_class(class_name)`: Lists supported talent builds for a class

## Base Combat Module

The `BaseCombatModule` abstract class defines the interface all class modules must implement:

1. Resource tracking (health, mana, rage, energy, focus)
2. Cooldown management
3. Buff and debuff tracking
4. Common utilities for distance calculation, target assessment, etc.

### Abstract Methods

Each class implementation must provide these methods:

- `get_optimal_rotation(state)`: Determines the best ability sequence
- `get_optimal_target(state)`: Selects the best target
- `get_optimal_position(state)`: Determines the best position in combat
- `get_resource_abilities(state)`: Returns resource-generating abilities
- `get_defensive_abilities(state)`: Returns defensive/survival abilities

## Class-Specific Modules

Each class has a specialized module in the `classes/` directory:

- `warrior.py`: Warrior implementation (Arms, Fury, Protection)
- `mage.py`: Mage implementation (Frost, Fire, Arcane)
- `priest.py`: Priest implementation (Shadow, Discipline, Holy)
- `hunter.py`: Hunter implementation (Beast Mastery, Marksmanship, Survival)
- And more...

### Specialization Handling

Each class module supports multiple specializations through dedicated rotation methods:

```python
def get_optimal_rotation(self, state):
    spec = self._determine_specialization(state)
    
    if spec == "arms":
        return self._get_arms_rotation(state)
    elif spec == "fury":
        return self._get_fury_rotation(state)
    elif spec == "protection":
        return self._get_protection_rotation(state)
    else:
        return self._get_leveling_rotation(state)
```

## Combat Situational Awareness

The `CombatSituationalAwareness` class provides tactical analysis of combat situations:

1. AOE detection
2. Interrupt target prioritization
3. Crowd control management
4. PvP vs PvE strategy adaptation
5. Group role coordination

### Key Features

- **Enemy Clustering**: Identifies groups of enemies for AoE abilities
- **Cast Analysis**: Detects which enemy spells should be interrupted
- **Danger Assessment**: Evaluates the threat level of enemy abilities
- **Tactical Suggestions**: Generates tactical recommendations

## Combat Flow

When the main system needs combat decisions, the following flow occurs:

1. Game state is provided to the `CombatManager`
2. Manager identifies the player's class and loads the appropriate module
3. `CombatSituationalAwareness` analyzes the tactical situation
4. Class module generates a rotation based on spec and situation
5. Class module selects target and positioning
6. Manager returns a prioritized list of combat actions

## Rotation System

Rotations are generated as priority lists, allowing the system to adapt to changing combat conditions:

```python
[
    {
        "name": "Shield Slam",
        "target": "target_id",
        "priority": 90,
        "resource_cost": 15,
        "condition": "not on cooldown"
    },
    {
        "name": "Revenge",
        "target": "target_id",
        "priority": 80,
        "resource_cost": 10,
        "condition": "proc active"
    },
    # More abilities...
]
```

The action system executes the highest priority ability that meets its conditions.

## Class-Specific Details

### Warrior

The Warrior module supports all three specializations with appropriate rotations:

- **Arms**: Focuses on Mortal Strike, Execute, and burst damage
- **Fury**: Dual-wield focused with sustained DPS via Bloodthirst and Whirlwind
- **Protection**: Tank spec using Shield Slam and defensive abilities

Key features:
- Stance management
- Rage optimization
- Execute phase detection

### Mage

The Mage module implements all three specializations:

- **Frost**: Control-oriented with Frostbolt, Ice Lance, and Frozen Orb
- **Fire**: Burst damage with Fireball, Pyroblast, and Hot Streak management
- **Arcane**: Mana management with Arcane Blast, Arcane Barrage cycle

Key features:
- Proc tracking (Hot Streak, Brain Freeze, etc.)
- Mana management
- Distance optimization

### Priest

The Priest module supports healing and damage specializations:

- **Shadow**: DoT management with Shadow Word: Pain, Vampiric Touch
- **Discipline**: Shield-focused with damage and healing
- **Holy**: Pure healing with single target and group heals

Key features:
- Form management (Shadowform)
- Target priority based on health
- Group healing coordination

## Testing Combat Modules

The `CombatTester` class provides tools for evaluating and refining combat modules:

1. Simulates combat scenarios
2. Measures DPS and resource usage
3. Compares different rotations
4. Handles AOE and multiple target scenarios

## Using the Combat System

To use the combat system in your code:

```python
from src.decision.combat_manager import CombatManager

# Initialize
combat_manager = CombatManager(config, knowledge)

# Generate a combat plan
game_state = perception.get_game_state()
combat_plan = combat_manager.generate_combat_plan(game_state)

# Execute the plan
for action in combat_plan:
    action_system.execute_action(action)
```

## Extending the Combat System

To add support for a new class:

1. Create a new file in `src/decision/combat/classes/`
2. Implement the `BaseCombatModule` interface
3. Add the class to imports in `classes/__init__.py`
4. Register the class in `combat_manager.py`

## Configuration Options

Combat behavior can be configured through `config.json`:

```json
{
  "combat": {
    "global": {
      "health_threshold": 30,
      "resource_threshold": 20,
      "aoe_threshold": 3
    },
    "warrior": {
      "preferred_spec": "arms",
      "stance_switching": true
    },
    "mage": {
      "preferred_spec": "frost",
      "distance_preference": 35
    }
  }
}
```

## Troubleshooting

Common issues and solutions:

1. **No combat actions generated**
   - Check if the player class is supported
   - Verify that a target is selected
   - Check for appropriate resources (mana, rage, etc.)

2. **Suboptimal ability usage**
   - Examine the rotation priority order
   - Check cooldown tracking
   - Verify debuff tracking functionality

3. **Missing specialized abilities**
   - Ensure the specialization detection is working
   - Check that abilities are properly registered

## Performance Considerations

The combat system is optimized for real-time decision making:

1. **Caching**: Combat modules are cached for performance
2. **Prioritization**: Only top actions are returned for execution
3. **Lazy Loading**: Class modules are loaded on demand