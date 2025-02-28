"""
Logger Module

Provides a centralized logging system for the application with
configurable log levels, rotation, and output formatting.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional


class LoggerSetup:
    """
    Sets up application-wide logging with configurable outputs
    and formats based on configuration settings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logger setup with the provided configuration
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.log_dir = self._get_log_dir()
        self.log_level = self._get_log_level()
        self.rotation_settings = self._get_rotation_settings()
        
        # Create root logger
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers (for reinitialization)
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        
        # Add handlers
        self._setup_console_handler()
        self._setup_file_handler()
        
        # Log setup completion
        logging.info(f"Logging system initialized at level {self.log_level}")
    
    def _get_log_dir(self) -> str:
        """
        Get the log directory from config or use default
        
        Returns:
            str: Path to log directory
        """
        log_dir = self.config.get("logging", {}).get("log_dir", "logs")
        
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        return log_dir
    
    def _get_log_level(self) -> int:
        """
        Get the log level from config or use default
        
        Returns:
            int: Logging level as defined in the logging module
        """
        level_name = self.config.get("logging", {}).get("log_level", "INFO")
        
        # Map level name to logging constant
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        return level_map.get(level_name.upper(), logging.INFO)
    
    def _get_rotation_settings(self) -> Dict[str, Any]:
        """
        Get the log rotation settings from config or use defaults
        
        Returns:
            Dict: Log rotation settings
        """
        rotation_config = self.config.get("logging", {}).get("rotation", {})
        
        return {
            "max_bytes": rotation_config.get("max_bytes", 10 * 1024 * 1024),  # 10 MB
            "backup_count": rotation_config.get("backup_count", 10),
            "when": rotation_config.get("when", "midnight"),
            "interval": rotation_config.get("interval", 1)
        }
    
    def _setup_console_handler(self) -> None:
        """Set up logging to console"""
        # Only add console handler if configured or in debug mode
        if self.config.get("logging", {}).get("console_output", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            # Create formatter
            formatter = self._create_formatter(colored=True)
            console_handler.setFormatter(formatter)
            
            # Add handler
            self.root_logger.addHandler(console_handler)
    
    def _setup_file_handler(self) -> None:
        """Set up logging to rotating file"""
        # Get log file path
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"claudewow_{timestamp}.log")
        
        # Create rotating file handler
        if self.config.get("logging", {}).get("rotation", {}).get("time_based", True):
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when=self.rotation_settings["when"],
                interval=self.rotation_settings["interval"],
                backupCount=self.rotation_settings["backup_count"]
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.rotation_settings["max_bytes"],
                backupCount=self.rotation_settings["backup_count"]
            )
        
        file_handler.setLevel(self.log_level)
        
        # Create formatter (no colors for file output)
        formatter = self._create_formatter(colored=False)
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.root_logger.addHandler(file_handler)
    
    def _create_formatter(self, colored: bool = False) -> logging.Formatter:
        """
        Create a formatter for log messages
        
        Args:
            colored: Whether to include ANSI color codes
            
        Returns:
            logging.Formatter: Configured formatter
        """
        if colored and sys.stdout.isatty():
            # Color codes
            colors = {
                'DEBUG': '\033[36m',     # Cyan
                'INFO': '\033[32m',      # Green
                'WARNING': '\033[33m',   # Yellow
                'ERROR': '\033[31m',     # Red
                'CRITICAL': '\033[41m',  # White on red background
                'RESET': '\033[0m'       # Reset
            }
            
            # Format with colors
            format_str = (
                "%(asctime)s - "
                "%(color)s%(levelname)-8s%(reset)s - "
                "%(name)s - "
                "%(message)s"
            )
            
            # Custom formatter to insert color codes
            class ColorFormatter(logging.Formatter):
                def format(self, record):
                    record.color = colors.get(record.levelname, '')
                    record.reset = colors['RESET']
                    return super().format(record)
            
            return ColorFormatter(format_str)
        else:
            # Standard format without colors
            format_str = (
                "%(asctime)s - "
                "%(levelname)-8s - "
                "%(name)s - "
                "%(message)s"
            )
            return logging.Formatter(format_str)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the given name
        
        Args:
            name: Logger name
            
        Returns:
            logging.Logger: Configured logger instance
        """
        return logging.getLogger(name)


# Singleton instance of the logger setup
_logger_setup = None


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Initialize the logging system with the given configuration
    
    Args:
        config: Application configuration
    """
    global _logger_setup
    _logger_setup = LoggerSetup(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if _logger_setup is None:
        # Initialize with default config if not already set up
        setup_logging({})
    
    return _logger_setup.get_logger(name)


# Performance logging utilities
class PerformanceTracker:
    """
    Utility class to track and log performance metrics
    """
    
    def __init__(self, module_name: str):
        """
        Initialize performance tracker
        
        Args:
            module_name: Name of the module being tracked
        """
        self.logger = get_logger(f"performance.{module_name}")
        self.module_name = module_name
        self.start_times = {}
    
    def start_tracking(self, operation: str) -> None:
        """
        Start tracking an operation's performance
        
        Args:
            operation: Name of the operation
        """
        self.start_times[operation] = datetime.now()
    
    def end_tracking(self, operation: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
        """
        End tracking an operation and log its duration
        
        Args:
            operation: Name of the operation
            extra_info: Additional information to log
            
        Returns:
            float: Duration in milliseconds
        """
        if operation not in self.start_times:
            self.logger.warning(f"Cannot end tracking for unknown operation: {operation}")
            return 0.0
        
        duration = (datetime.now() - self.start_times[operation]).total_seconds() * 1000
        
        # Log performance
        log_msg = f"{operation} completed in {duration:.2f}ms"
        if extra_info:
            log_msg += f" | {extra_info}"
        
        # Log at different levels based on duration
        if duration > 1000:  # > 1 second
            self.logger.warning(log_msg)
        else:
            self.logger.debug(log_msg)
        
        # Clean up
        del self.start_times[operation]
        
        return duration