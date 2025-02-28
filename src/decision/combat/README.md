# Combat System for ClaudeWoW

This directory contains the class-specific combat modules and related components for the World of Warcraft AI system.

## Overview

The combat system provides specialized combat logic for different World of Warcraft classes, handling:

- Rotation priorities
- Resource management (mana, rage, energy, focus)
- Cooldown tracking
- Situational awareness
- Positioning logic
- Defensive ability usage
- Target selection

## Components

### Base Architecture

- `base_combat_module.py`: The abstract base class defining the interface for all class modules
- `combat_manager.py`: Primary interface for the rest of the system to interact with combat logic
- `situational_awareness.py`: Provides combat situational analysis capabilities

### Class-Specific Implementations

All class implementations are in the `classes/` directory:

- `warrior.py`: Warrior implementation (Arms, Fury, Protection)
- `mage.py`: Mage implementation (Frost, Fire, Arcane)
- `priest.py`: Priest implementation (Shadow, Discipline, Holy)
- `hunter.py`: Hunter implementation (Beast Mastery, Marksmanship, Survival)
- `rogue.py`: Rogue implementation (Combat, Assassination, Subtlety)
- `warlock.py`: Warlock implementation (Affliction, Demonology, Destruction)
- `paladin.py`: Paladin implementation (Holy, Protection, Retribution)
- `shaman.py`: Shaman implementation (Elemental, Enhancement, Restoration)
- `druid.py`: Druid implementation (Balance, Feral, Restoration)

### Testing and Analysis

- `combat_testing.py`: Test harness for evaluating combat module performance

## Combat Situational Awareness

The situational awareness system provides tactical analysis for combat:

- AOE detection and targeting
- Interrupt priorities
- Crowd control management
- PvP vs PvE strategy switching
- Group role awareness (tank, healer, DPS)
- Danger assessment

## Usage

The combat system is primarily interfaced through the `CombatManager` class:

```python
# Initialize the combat manager
combat_manager = CombatManager(config, knowledge)

# Generate a combat plan based on current game state
combat_plan = combat_manager.generate_combat_plan(state)

# Execute actions from the combat plan
for action in combat_plan:
    # Execute action
    # ...
```

## Testing

The `CombatTester` class can be used to evaluate combat module performance:

```python
# Initialize the tester
tester = CombatTester(config, knowledge)

# Set up a test
tester.setup_test("Warrior DPS Test", {
    "player_class": "warrior",
    "warrior_spec": "arms",
    "player_level": 60
}, duration=120.0)

# Run the test
results = tester.run_test()

# Save test results
tester.save_results()
```

## Extending

To add a new class implementation:

1. Create a new file in `classes/` (e.g., `death_knight.py`)
2. Implement the `BaseCombatModule` interface
3. Add the class to the imports in `classes/__init__.py`
4. Register the class in `combat_manager.py`