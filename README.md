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

## System Architecture

ClaudeWoW uses a perception-decision-action architecture:

- **Perception**: Screen reading, OCR, entity detection, minimap analysis
- **Decision**: Planning, navigation, combat AI, quest management
- **Action**: Game control, ability execution, movement

## Learning Capabilities

- **Reinforcement Learning**: Q-learning with experience buffer
- **Knowledge Expansion**: Pattern detection and knowledge base expansion
- **Performance Metrics**: Compare against human benchmarks
- **Transfer Learning**: Apply skills across different contexts
- **Hierarchical Planning**: Sophisticated goal management

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

3. **Next Phases** (See NEXT_STEPS.md for details)
   - Class-specific combat modules (2-3 weeks)
   - Economic intelligence system (2-3 weeks)
   - Enhanced learning capabilities (3-4 weeks)
   - Social intelligence system (2-3 weeks)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.