# ClaudeWoW Documentation

Welcome to the ClaudeWoW documentation. This directory contains detailed information about the system architecture, components, and how they work together.

## Table of Contents

- [Architecture Overview](architecture.md) - System architecture and component interactions
- [Navigation System](navigation.md) - Advanced 3D pathfinding and navigation
- [Learning System](learning.md) - Learning capabilities and knowledge expansion
- [Combat AI](combat.md) - Combat decision making and class rotations
- [Perception System](perception.md) - Screen reading and game state detection
- [Action System](action.md) - Game control and action execution

## Project Structure

ClaudeWoW follows a perception-decision-action architecture:

```
src/
├── perception/ - Game state detection
│   ├── screen_reader.py - Basic screen reading
│   ├── entity_detector.py - Entity detection
│   ├── minimap_analyzer.py - Minimap analysis
│   ├── text_extractor.py - Text extraction (OCR)
│   └── ui_detector.py - UI element detection
│
├── decision/ - Decision making
│   ├── agent.py - Main agent decision making
│   ├── behavior_tree.py - Behavior tree implementation
│   ├── combat_manager.py - Combat decision making
│   ├── navigation_manager.py - Basic navigation
│   ├── advanced_navigation.py - 3D navigation
│   ├── advanced_pathfinding.py - Advanced pathfinding algorithms
│   ├── terrain_analyzer.py - Terrain analysis
│   ├── flight_path_manager.py - Flight path management
│   ├── dungeon_navigator.py - Dungeon navigation
│   ├── planner.py - High-level planning
│   └── quest_manager.py - Quest tracking and management
│
├── learning/ - Learning systems
│   ├── reinforcement_learning.py - RL implementation
│   ├── knowledge_expansion.py - Knowledge expansion
│   ├── performance_metrics.py - Performance tracking
│   ├── transfer_learning.py - Skill transfer
│   └── hierarchical_planning.py - HTN planning
│
├── action/ - Action execution
│   ├── controller.py - Raw input control
│   └── movement_controller.py - Movement execution
│
├── knowledge/ - Game knowledge
│   └── game_knowledge.py - Game knowledge base
│
├── social/ - Social interaction
│   ├── chat_analyzer.py - Chat analysis
│   ├── llm_interface.py - LLM interface
│   └── social_manager.py - Social interaction management
│
└── utils/ - Utilities
    ├── config.py - Configuration utilities
    └── gui_overlay.py - GUI overlay for debugging
```

## Getting Started

To get started with ClaudeWoW:

1. Make sure you have Python 3.9+ installed
2. Clone the repository
3. Install dependencies: `pip install -r requirements.txt`
4. Run initial setup: `python initial_setup.ps1`
5. Start the AI: `python main.py`

## Configuration

The main configuration file is `config/config.json`, which controls behavior of all components. See the [Configuration Guide](configuration.md) for details.

## Contributing

Contributions are welcome! Please see [Contributing Guidelines](../CONTRIBUTING.md) for details on how to contribute to the project.

## Current Status and Roadmap

See [NEXT_STEPS.md](../NEXT_STEPS.md) for the current project status and development roadmap.