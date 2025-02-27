# WoW AI Player Project Setup Script
# This script creates the project folder structure and initializes the Python environment

# Define project root directory (change as needed)
$projectRoot = "D:\Repos\ClaudeWoW"

# Create main project directory if it doesn't exist
if (-not (Test-Path $projectRoot)) {
    New-Item -ItemType Directory -Path $projectRoot
    Write-Host "Created project root directory: $projectRoot" -ForegroundColor Green
}

# Define folder structure
$folders = @(
    "src",
    "src\perception",
    "src\decision",
    "src\action",
    "src\knowledge",
    "src\learning",
    "src\utils",
    "data",
    "data\recordings",
    "data\models",
    "data\game_knowledge",
    "config",
    "logs",
    "tests"
)

# Create folders
foreach ($folder in $folders) {
    $path = Join-Path -Path $projectRoot -ChildPath $folder
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path
        Write-Host "Created directory: $path" -ForegroundColor Green
    }
}

# Create Python virtual environment
Set-Location $projectRoot
Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
python -m venv venv
Write-Host "Virtual environment created." -ForegroundColor Green

# Activate virtual environment and install dependencies
Write-Host "Installing required packages..." -ForegroundColor Yellow
& "$projectRoot\venv\Scripts\activate.ps1"

# Install required packages
$packages = @(
    "numpy",
    "opencv-python",
    "pytesseract",
    "torch",
    "tensorflow",
    "stable-baselines3",
    "gymnasium",
    "pyautogui",
    "pynput",
    "networkx",
    "pandas",
    "matplotlib",
    "pytest",
    "pylint",
    "black",
    "jupyter"
)

foreach ($package in $packages) {
    pip install $package
    Write-Host "Installed $package" -ForegroundColor Green
}

# Create requirements.txt
pip freeze > requirements.txt
Write-Host "Created requirements.txt" -ForegroundColor Green

# Create basic config file
$configContent = @"
{
    "game_path": "C:\\Program Files (x86)\\World of Warcraft\\_retail_\\Wow.exe",
    "screenshot_interval": 0.1,
    "input_delay": 0.05,
    "log_level": "INFO",
    "ui_scale": 1.0,
    "resolution": {
        "width": 1920,
        "height": 1080
    },
    "model_paths": {
        "vision": "data/models/vision_model.pt",
        "combat": "data/models/combat_model.pt",
        "navigation": "data/models/navigation_model.pt"
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.0001,
        "epochs": 100
    }
}
"@

$configPath = Join-Path -Path $projectRoot -ChildPath "config\config.json"
Set-Content -Path $configPath -Value $configContent
Write-Host "Created basic configuration file" -ForegroundColor Green

# Create main.py
$mainPyContent = @"
"""
WoW AI Player - Main Entry Point
"""
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.screen_reader import ScreenReader
from src.decision.agent import Agent
from src.action.controller import Controller
from src.utils.config import load_config

def setup_logging():
    """Configure logging"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"wow_ai_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("wow_ai")

def main():
    """Main entry point for the WoW AI Player"""
    logger = setup_logging()
    logger.info("Starting WoW AI Player")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    try:
        # Initialize components
        screen_reader = ScreenReader(config)
        controller = Controller(config)
        agent = Agent(config, screen_reader, controller)
        
        logger.info("All components initialized successfully")
        
        # Main loop
        logger.info("Entering main loop")
        while True:
            # Process game state
            game_state = screen_reader.capture_game_state()
            
            # Decide on actions
            actions = agent.decide(game_state)
            
            # Execute actions
            controller.execute(actions)
            
            # Sleep to prevent high CPU usage
            time.sleep(config.get("loop_interval", 0.1))
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
    finally:
        logger.info("WoW AI Player shutting down")

if __name__ == "__main__":
    main()
"@

$mainPyPath = Join-Path -Path $projectRoot -ChildPath "main.py"
Set-Content -Path $mainPyPath -Value $mainPyContent
Write-Host "Created main.py entry point" -ForegroundColor Green

# Create basic README.md
$readmeContent = @"
# WoW AI Player

An AI system designed to autonomously play World of Warcraft in a sandbox environment.

## Project Structure

- `src/` - Source code
  - 'perception/' - Screen reading and game state interpretation
  - 'decision/' - AI decision-making components
  - 'action/' - Game interaction and control systems
  - 'knowledge/' - Game knowledge representation
  - 'learning/' - Machine learning models and training code
  - 'utils/' - Utility functions and helpers
- `data/` - Data storage
  - `recordings/` - Gameplay recordings for training
  - `models/` - Trained ML models
  - `game_knowledge/` - Game data and knowledge base
- `config/` - Configuration files
- `logs/` - Application logs
- `tests/` - Unit and integration tests

## Setup

1. Run `initial_setup.ps1` to create the project structure and install dependencies
2. Modify `config/config.json` with your specific game installation path and preferences
3. Run `main.py` to start the AI

## Requirements

- Python 3.8+
- World of Warcraft installed
- CUDA-compatible GPU (recommended for training)

## License

This project is for educational purposes only. Use in accordance with Blizzard's Terms of Service.
"@

$readmePath = Join-Path -Path $projectRoot -ChildPath "README.md"
Set-Content -Path $readmePath -Value $readmeContent
Write-Host "Created README.md" -ForegroundColor Green

# Create .gitignore
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Logs
logs/
*.log

# Data
data/recordings/
data/models/
*.pkl
*.h5
*.pt
*.pth

# IDEs
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"@

$gitignorePath = Join-Path -Path $projectRoot -ChildPath ".gitignore"
Set-Content -Path $gitignorePath -Value $gitignoreContent
Write-Host "Created .gitignore" -ForegroundColor Green

# Final message
Write-Host "`nWoW AI Player project structure setup complete!" -ForegroundColor Cyan
Write-Host "Project located at: $projectRoot" -ForegroundColor Cyan
Write-Host "To start working on the project:" -ForegroundColor Cyan
Write-Host "1. cd $projectRoot" -ForegroundColor White
Write-Host "2. .\venv\Scripts\activate" -ForegroundColor White
Write-Host "3. python main.py (once you've implemented the required components)" -ForegroundColor White
Write-Host "`nHappy coding!" -ForegroundColor Cyan