"""
Configuration Utility

This module handles loading and validating configuration settings.
"""

import logging
import json
import os
import sys
from typing import Dict, Any, Optional

# Setup module-level logger
logger = logging.getLogger("wow_ai.utils.config")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file
    
    Args:
        config_path: Path to the config file. If None, uses default location.
    
    Returns:
        Dict: Configuration dictionary
    """
    # Get default config path if not provided
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "config.json"
        )
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        # Validate the configuration
        validate_config(config)
        
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return create_default_config(config_path)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file {config_path}")
        return create_default_config(config_path)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return create_default_config(config_path)

def create_default_config(config_path: str) -> Dict[str, Any]:
    """
    Create and save a default configuration
    
    Args:
        config_path: Path where to save the default config
    
    Returns:
        Dict: Default configuration dictionary
    """
    default_config = {
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
        },
        "tesseract_path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        "debug": {
            "save_screenshots": False,
            "ocr_debug": False,
            "vision_debug": False
        }
    }
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save default config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        logger.info(f"Created default configuration at {config_path}")
    except Exception as e:
        logger.error(f"Error creating default configuration: {e}")
    
    return default_config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration settings
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Required top-level keys
    required_keys = ["resolution", "game_path", "screenshot_interval", "input_delay"]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            logger.warning(f"Missing required configuration key: {key}")
            return False
    
    # Validate resolution
    if "resolution" in config:
        if not isinstance(config["resolution"], dict):
            logger.warning("Resolution setting must be a dictionary")
            return False
        
        if "width" not in config["resolution"] or "height" not in config["resolution"]:
            logger.warning("Resolution setting must include width and height")
            return False
        
        if not isinstance(config["resolution"]["width"], int) or not isinstance(config["resolution"]["height"], int):
            logger.warning("Resolution width and height must be integers")
            return False
    
    # Validate intervals
    if "screenshot_interval" in config and not isinstance(config["screenshot_interval"], (int, float)):
        logger.warning("Screenshot interval must be a number")
        return False
    
    if "input_delay" in config and not isinstance(config["input_delay"], (int, float)):
        logger.warning("Input delay must be a number")
        return False
    
    return True

def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Save configuration to a JSON file
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the config file. If None, uses default location.
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    # Get default config path if not provided
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "config.json"
        )
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def update_config(updates: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Update configuration with new values
    
    Args:
        updates: Dictionary of settings to update
        config_path: Path to the config file. If None, uses default location.
    
    Returns:
        Dict: Updated configuration dictionary
    """
    # Load existing config
    config = load_config(config_path)
    
    # Update settings
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
    
    deep_update(config, updates)
    
    # Save updated config
    save_config(config, config_path)
    
    return config