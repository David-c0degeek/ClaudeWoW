# WoW AI Player

An advanced AI system designed to autonomously play World of Warcraft in a private sandbox environment.

![WoW AI Player](docs/images/header_image.png)

## Overview

This project implements a complete AI player for World of Warcraft that can:

- Perceive and understand the game state through computer vision
- Make intelligent decisions based on game knowledge and current state
- Execute actions through keyboard and mouse control
- Learn and improve over time
- Navigate the game world and complete quests
- Engage in combat with appropriate class-specific rotations
- Adapt to different situations and challenges

**Important Note:** This project is intended for use in a private sandbox environment with proper permissions. Botting on official World of Warcraft servers is against the Terms of Service.

## Architecture

The system is structured into three main components:

### 1. Perception System
- Screen analysis to extract game state information
- UI element detection and OCR for text reading
- Entity detection and tracking
- Minimap analysis for navigation

### 2. Decision System
- Behavior tree for high-level decision making
- Specialized managers for combat, navigation, and quests
- Planning and goal management
- Knowledge base of game information

### 3. Action System
- Keyboard and mouse control
- Action execution and timing
- Movement and targeting
- Spell casting and ability usage

## Requirements

- Python 3.8+
- A local World of Warcraft installation
- CUDA-compatible GPU (recommended for faster vision processing)
- Dependencies:
  - OpenCV
  - PyTorch/TensorFlow
  - Tesseract OCR
  - pyautogui/pynput
  - And more (see `requirements.txt`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/wow-ai-player.git
cd wow-ai-player
```

2. Run the setup script to create the folder structure and install dependencies:
```bash
# On Windows
powershell -ExecutionPolicy Bypass -File initial_setup.ps1

# On Linux/Mac
# First make the script executable
chmod +x initial_setup.sh
./initial_setup.sh
```

3. Install Tesseract OCR:
   - Windows: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

4. Update the configuration file with your specific settings:
```bash
# Edit config/config.json with your preferred editor
# Make sure to set the correct game path and resolution
```

## Usage

1. Launch the WoW AI player:
```bash
# Basic usage
python launcher.py

# With specific config file
python launcher.py --config path/to/config.json

# With startup delay (to give you time to switch to game window)
python launcher.py --delay 5

# With debug logging
python launcher.py --log-level DEBUG
```

2. The AI will begin analyzing the screen and making decisions once the game is running.

3. Press Ctrl+C in the terminal to stop the AI.

## Configuration

The system is highly configurable through the `config/config.json` file:

```json
{
  "game_path": "C:\\Program Files (x86)\\World of Warcraft\\_retail_\\Wow.exe",
  "screenshot_interval": 0.1,
  "input_delay": 0.05,
  "log_level": "INFO",
  "ui_scale": 1.0,
  "resolution": {
    "width": 1920,
    "height": 1080
  }
}
```

Key configuration options:
- `game_path`: Path to your WoW executable
- `screenshot_interval`: How frequently to capture the screen (in seconds)
- `input_delay`: Delay between inputs to avoid overwhelming the game
- `ui_scale`: Your in-game UI scale setting
- `resolution`: Your game resolution

## Extending the System

### Adding New Abilities

Add new class abilities to `data/game_knowledge/abilities.json`:

```json
{
  "warrior": {
    "new_ability": {
      "name": "New Ability",
      "rank": 1,
      "level": 10,
      "type": "ability",
      "resource": "rage",
      "cost": 20,
      "cooldown": 30,
      "effects": [
        {"type": "damage", "target": "enemy", "value": 50}
      ]
    }
  }
}
```

### Adding Quest Information

Add new quests to `data/game_knowledge/quests.json`:

```json
{
  "new_quest_id": {
    "id": "new_quest_id",
    "title": "New Quest Title",
    "level": 10,
    "faction": "alliance",
    "zone": "elwynn_forest",
    "quest_giver": "npc_id",
    "turn_in": "npc_id",
    "objectives": [
      {
        "type": "kill",
        "target": "mob_name",
        "count": 10,
        "description": "Kill 10 Wolves"
      }
    ]
  }
}
```

### Training Custom Vision Models

To improve vision capabilities:

1. Collect training data by running in data collection mode:
```bash
python tools/collect_training_data.py --output data/training
```

2. Train a new model:
```bash
python tools/train_vision_model.py --data data/training --output data/models/vision_model.pt
```

3. Update the config to use your new model:
```json
{
  "model_paths": {
    "vision": "data/models/vision_model.pt"
  }
}
```

## LLM Integration for Social Intelligence

The WoW AI player includes advanced social intelligence powered by Large Language Models (LLMs). This enables natural conversations with other players and realistic social behaviors.

### Supported LLM Providers

The system supports multiple LLM providers:

- **OpenAI (GPT models)**: High-quality responses with excellent gaming knowledge
- **Anthropic (Claude models)**: Natural conversational abilities with strong role-playing
- **Google (Gemini models)**: Google's AI models with strong general knowledge
- **Azure OpenAI**: Enterprise-grade OpenAI models with added security
- **Mistral AI**: Cost-effective alternative with competitive performance
- **Ollama**: Local API for running open models on your machine
- **Local Models**: Completely offline models for maximum privacy

### Configuration

You can configure LLM settings through the GUI launcher or by editing `config/llm_config.json`:

1. Launch the configuration GUI:
   ```bash
   python launcher.py --gui

## Project Structure

```
wow-ai-player/
├── config/                # Configuration files
│   └── config.json        # Main configuration
├── data/                  # Data storage
│   ├── game_knowledge/    # Game information database
│   ├── models/            # Trained ML models
│   ├── recordings/        # Gameplay recordings for training
│   └── templates/         # Template images for matching
├── logs/                  # Application logs
├── src/                   # Source code
│   ├── action/            # Action execution
│   ├── decision/          # Decision making
│   ├── knowledge/         # Knowledge representation
│   ├── learning/          # Machine learning
│   ├── perception/        # Game state perception
│   └── utils/             # Utility functions
├── tests/                 # Unit and integration tests
├── tools/                 # Utility scripts
├── initial_setup.ps1      # Windows setup script
├── initial_setup.sh       # Linux/Mac setup script
├── launcher.py            # Main entry point
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## License

This project is for educational purposes only. Use in accordance with Blizzard's Terms of Service.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Disclaimer

This software is intended for educational purposes and private use in sandbox environments only. The authors do not condone or support:

1. Using this software on official World of Warcraft servers
2. Violating Blizzard's Terms of Service
3. Gaining unfair advantages in multiplayer environments
4. Any commercial use of this software

The authors are not responsible for any misuse of this software or any consequences resulting from such misuse.