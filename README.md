# ClaudeWoW

An autonomous AI for playing World of Warcraft with advanced learning and planning capabilities.

## Advanced Pathfinding and Navigation

ClaudeWoW's advanced pathfinding and navigation system provides sophisticated movement capabilities:

### Features

- **3D Pathfinding**: Full support for elevation and vertical movement
- **Terrain Analysis**: Automatically analyzes terrain difficulty and traversability
- **Multi-zone Routing**: Navigate seamlessly between different zones
- **Dungeon Navigation**: Specialized pathfinding for dungeon environments
- **Flight Path Integration**: Intelligent use of in-game flight paths

### Components

- `advanced_navigation.py` - Core 3D navigation with terrain handling
- `advanced_pathfinding.py` - Multiple pathfinding algorithms (A*, JPS, Theta*, RRT)
- `terrain_analyzer.py` - Terrain classification and obstacle detection
- `flight_path_manager.py` - Flight path optimization and routing
- `dungeon_navigator.py` - Specialized dungeon navigation

### Pathfinding Algorithms

- **A*** - Standard pathfinding for general navigation
- **Jump Point Search (JPS)** - Optimized for grid-based environments
- **Theta*** - For smooth, any-angle paths
- **RRT** - For complex 3D environments with narrow passages

## Economic Intelligence System

ClaudeWoW's economic intelligence system provides sophisticated market analysis and decision-making capabilities:

### Features

- **Market Analysis**: Auction house data collection, trend detection, and price prediction
- **Farming Optimization**: Resource node mapping and efficient farming route calculation
- **Crafting Intelligence**: Profitability calculations, material sourcing, and recipe prioritization
- **Inventory Management**: Value-based inventory decisions and gold optimization

### Components

- `market_analyzer.py` - Auction house data processing and price prediction
- `farming_optimizer.py` - Resource node tracking and route optimization
- `crafting_manager.py` - Profitability analysis and crafting decisions
- `inventory_manager.py` - Inventory organization and decision-making
- `economic_manager.py` - Integrated economic system coordinator

### Capabilities

- Calculate optimal farming routes based on profession skills and market prices
- Generate crafting shopping lists with material sourcing recommendations
- Analyze inventory and recommend items to vendor, auction, or keep
- Create comprehensive gold-making plans with prioritized recommendations

## Class-Specific Combat System

ClaudeWoW's combat system provides specialized mechanics for each WoW class:

### Features

- **Class Framework**: Base combat module with rotation priority management
- **Resource Systems**: Specialized resource handling (mana, rage, energy, focus)
- **Rotation Intelligence**: Optimal ability sequencing based on situation 
- **Combat Awareness**: AOE detection, interrupt priorities, and situational tactics

### Components

- `combat_manager.py` - Primary combat decision engine
- `base_combat_module.py` - Framework for all class modules
- `classes/*.py` - Class-specific implementations
- `situational_awareness.py` - Combat situation analysis

### Capabilities

- Specialized rotations for all WoW classes and talent specializations
- Dynamic target selection and positioning logic
- Tactical awareness for PvP vs PvE contexts
- Group role coordination (tank, healer, DPS)

## System Architecture

ClaudeWoW uses a perception-decision-action architecture:

- **Perception**: Screen reading, OCR, entity detection, minimap analysis
- **Decision**: Planning, navigation, combat AI, quest management
- **Action**: Game control, ability execution, movement

## Learning Capabilities

- **Deep Reinforcement Learning**: Neural networks for complex decision making
- **Imitation Learning**: Learn patterns from expert human gameplay
- **Visual Processing**: Screen understanding for navigation and combat
- **Multi-Agent Learning**: Group coordination strategies
- **Knowledge Expansion**: Pattern detection and knowledge base extension
- **Hierarchical Planning**: Sophisticated goal management

## Social Intelligence

ClaudeWoW includes comprehensive social interaction capabilities:

- **Chat Analysis**: Context-aware chat understanding and response generation
- **Group Coordination**: Team communication and role coordination 
- **Reputation Management**: Track and manage social relationships with players
- **Guild Interaction**: Appropriate guild participation and cooperation
- **Scam Detection**: Identify and avoid potentially harmful interactions

## Getting Started

1. Install requirements: `pip install -r requirements.txt`
2. Run initial setup: `python initial_setup.ps1`
3. Start the AI: `python main.py`

## Configuration

Edit `config/config.json` to customize AI behavior:

- Game settings (resolution, keybindings)
- AI behavior settings
- Learning parameters
- Navigation preferences

## Documentation

See the `docs/` directory for detailed documentation:

- [Architecture Overview](docs/architecture.md)
- [Navigation System](docs/navigation.md)
- [Learning System](docs/learning.md)
- [Combat AI](docs/combat.md)
- [Economic System](docs/economic.md)

## Requirements

- Python 3.9+
- OpenCV
- NumPy
- PyTorch
- Tesseract OCR

## Development Roadmap

1. ✅ **Integration & Testing Phase** (COMPLETED)
2. ✅ **Advanced Navigation System** (COMPLETED)
   - 3D pathfinding with terrain handling
   - Specialized algorithms for different terrain types
   - Flight path integration and optimization
   - Dungeon navigation capabilities
3. ✅ **Economic Intelligence System** (COMPLETED)
   - Market analysis and price prediction
   - Farming route optimization
   - Crafting profitability analysis
   - Inventory management
4. ✅ **Class-Specific Combat Modules** (COMPLETED)
   - Base combat module framework
   - Class-specific rotation systems
   - Resource management (mana, rage, energy, focus)
   - Combat situational awareness system
5. ✅ **Enhanced Learning Capabilities** (COMPLETED)
   - Deep reinforcement learning implementation
   - Imitation learning from player recordings
   - Visual processing for terrain navigation
   - Multi-agent learning for group coordination
6. ✅ **Social Intelligence System** (COMPLETED)
   - Context-aware chat processing
   - Guild and player reputation management
   - Trade partner trust assessment
   - Scam and harassment detection
7. **Next Phases** (See NEXT_STEPS.md for details)
   - Technical debt and infrastructure improvements
   - Comprehensive documentation

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.