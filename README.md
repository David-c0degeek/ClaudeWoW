# WoW AI Player

An AI system designed to autonomously play World of Warcraft in a sandbox environment.

## Project Structure

- src/ - Source code
  - 'perception/' - Screen reading and game state interpretation
  - 'decision/' - AI decision-making components
  - 'action/' - Game interaction and control systems
  - 'knowledge/' - Game knowledge representation
  - 'learning/' - Machine learning models and training code
  - 'utils/' - Utility functions and helpers
- data/ - Data storage
  - ecordings/ - Gameplay recordings for training
  - models/ - Trained ML models
  - game_knowledge/ - Game data and knowledge base
- config/ - Configuration files
- logs/ - Application logs
- 	ests/ - Unit and integration tests

## Setup

1. Run initial_setup.ps1 to create the project structure and install dependencies
2. Modify config/config.json with your specific game installation path and preferences
3. Run main.py to start the AI

## Requirements

- Python 3.8+
- World of Warcraft installed
- CUDA-compatible GPU (recommended for training)

## License

This project is for educational purposes only. Use in accordance with Blizzard's Terms of Service.
