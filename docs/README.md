# ClaudeWoW Documentation

Welcome to the ClaudeWoW documentation. This directory contains comprehensive 
documentation for the AI system designed to play World of Warcraft autonomously.

## Documentation Sections

### 1. Architecture and Design

- **[Architecture Overview](architecture.md)** - The high-level architecture and design of the system
- **[Code Structure](code_structure.md)** - How the code is organized and structured
- **[Configuration Guide](configuration.md)** - How to configure the system

### 2. Core Systems

- **[Navigation System](navigation.md)** - Detailed documentation of the pathfinding and navigation capabilities
- **[Combat System](combat.md)** - Combat decision making and class-specific implementations
- **[Economic System](economic.md)** - Market analysis, farming optimization, and inventory management
- **[Learning System](learning.md)** - Machine learning capabilities for improved performance
- **[Social System](social.md)** - Chat analysis and social interactions

### 3. Development Guides

- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute to the project
- **[Testing Guide](testing.md)** - How to write and run tests
- **[Performance Tuning](performance.md)** - Performance optimization guidelines

### 4. User Guides

- **[Installation Guide](installation.md)** - How to install and set up the system
- **[Usage Guide](usage.md)** - How to use the system
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Getting Started

If you're new to ClaudeWoW, we recommend starting with:

1. The [Architecture Overview](architecture.md) to understand the system design
2. The [Installation Guide](installation.md) to set up the system
3. The [Usage Guide](usage.md) to start using ClaudeWoW

## Core Components

ClaudeWoW consists of several interconnected systems:

### Perception System
Analyzes the game screen, extracts information, and builds a game state model.

### Decision System
Makes decisions based on the current game state, including navigation, combat, and economic decisions.

### Action System
Executes decisions by interacting with the game through keyboard and mouse inputs.

### Learning System
Improves performance over time through reinforcement learning and imitation learning.

### Knowledge System
Stores and retrieves game knowledge, such as quest information, item data, and terrain details.

## Advanced Documentation

For more detailed information on specific components, please refer to the respective documentation files listed above.

## API Reference

For developers looking to extend ClaudeWoW's functionality, comprehensive API documentation is available for each module in their respective source files.

## Ongoing Development

ClaudeWoW is under active development. For the latest roadmap and milestones, please refer to the [NEXT_STEPS.md](../NEXT_STEPS.md) file in the root directory.