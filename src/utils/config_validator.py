"""
Configuration Validator Module

Validates application configuration against schema, provides default values,
and handles configuration errors.
"""

import os
import json
import jsonschema
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates and normalizes application configuration against a schema.
    
    Ensures that all required configuration values are present and valid,
    applies default values for missing optional settings, and converts
    data types as needed.
    """
    
    def __init__(self, schema_path: str):
        """
        Initialize with a schema file
        
        Args:
            schema_path: Path to the JSON schema file
        """
        self.schema = self._load_schema(schema_path)
    
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """
        Load the schema from a JSON file
        
        Args:
            schema_path: Path to the schema file
            
        Returns:
            Dict: The loaded schema
        """
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load schema from {schema_path}: {e}")
            # Return a minimal schema to allow basic validation
            return {"type": "object", "properties": {}}
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration against the schema
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        error_messages = []
        
        try:
            jsonschema.validate(instance=config, schema=self.schema)
            return True, []
        except jsonschema.exceptions.ValidationError as e:
            # Extract validation error messages
            error_path = "/".join(str(p) for p in e.path)
            if error_path:
                error_messages.append(f"Error at '{error_path}': {e.message}")
            else:
                error_messages.append(f"Validation error: {e.message}")
            
            return False, error_messages
    
    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values from schema to missing configuration items
        
        Args:
            config: The configuration to update with defaults
            
        Returns:
            Dict: Configuration with defaults applied
        """
        # Make a copy to avoid modifying the original
        config_with_defaults = config.copy()
        
        # Apply defaults recursively
        self._apply_defaults_recursive(config_with_defaults, self.schema)
        
        return config_with_defaults
    
    def _apply_defaults_recursive(self, config: Dict[str, Any], schema: Dict[str, Any], 
                                 path: str = "") -> None:
        """
        Recursively apply defaults from schema to config
        
        Args:
            config: Configuration section to update
            schema: Schema section with defaults
            path: Current path for logging
        """
        # Only process object types
        if schema.get("type") != "object":
            return
        
        # Process properties
        for prop_name, prop_schema in schema.get("properties", {}).items():
            # Build the current path for this property
            current_path = f"{path}/{prop_name}" if path else prop_name
            
            # If property has a default and is missing from config, add it
            if "default" in prop_schema and prop_name not in config:
                config[prop_name] = prop_schema["default"]
                logger.debug(f"Applied default {prop_schema['default']} to {current_path}")
            
            # If property exists and is an object, recursively process it
            if (prop_name in config and 
                isinstance(config[prop_name], dict) and 
                prop_schema.get("type") == "object"):
                
                self._apply_defaults_recursive(config[prop_name], prop_schema, current_path)
    
    def validate_and_apply_defaults(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Validate and apply defaults to a configuration
        
        Args:
            config: Configuration to validate and update
            
        Returns:
            Tuple[Dict, bool, List]: (updated_config, is_valid, error_messages)
        """
        # First apply defaults
        config_with_defaults = self.apply_defaults(config)
        
        # Then validate
        is_valid, error_messages = self.validate(config_with_defaults)
        
        return config_with_defaults, is_valid, error_messages


class ConfigManager:
    """
    Manages loading, validation, and access to application configuration.
    """
    
    def __init__(self, config_path: str, schema_path: str):
        """
        Initialize with config and schema paths
        
        Args:
            config_path: Path to the config file
            schema_path: Path to the schema file
        """
        self.config_path = config_path
        self.validator = ConfigValidator(schema_path)
        self.config = self._load_and_validate_config()
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """
        Load and validate the configuration
        
        Returns:
            Dict: Validated configuration
        """
        # Load config
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            config = {}
        
        # Validate and apply defaults
        updated_config, is_valid, error_messages = self.validator.validate_and_apply_defaults(config)
        
        # Log validation results
        if is_valid:
            logger.info("Configuration validated successfully")
        else:
            for error in error_messages:
                logger.error(f"Configuration error: {error}")
        
        return updated_config
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the full validated configuration
        
        Returns:
            Dict: The configuration dictionary
        """
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Any: The configuration value
        """
        parts = key.split('.')
        
        # Navigate through the config
        current = self.config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def save_config(self) -> bool:
        """
        Save the current configuration to file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def update(self, key: str, value: Any) -> bool:
        """
        Update a configuration value and save
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
            
        Returns:
            bool: True if update was successful
        """
        parts = key.split('.')
        
        # Navigate to the right level
        current = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Update the value
        current[parts[-1]] = value
        
        # Validate the updated config
        updated_config, is_valid, error_messages = self.validator.validate_and_apply_defaults(self.config)
        
        if is_valid:
            self.config = updated_config
            return self.save_config()
        else:
            logger.error(f"Failed to update config - validation errors: {error_messages}")
            return False