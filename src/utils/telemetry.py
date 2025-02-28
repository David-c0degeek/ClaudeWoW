"""
Telemetry Module

Provides performance monitoring, metrics collection, and system health tracking.
"""

import time
import threading
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

# Configure logging
logger = logging.getLogger(__name__)


class Metric:
    """
    Base class for metrics that can be collected and tracked.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a metric
        
        Args:
            name: Metric name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary for serialization
        
        Returns:
            Dict: Metric as dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "last_updated": self.last_updated,
            "type": self.__class__.__name__
        }


class CounterMetric(Metric):
    """
    A metric that counts occurrences of an event.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a counter metric
        
        Args:
            name: Metric name
            description: Human-readable description
        """
        super().__init__(name, description)
        self.count = 0
    
    def increment(self, value: int = 1) -> None:
        """
        Increment the counter by the specified value
        
        Args:
            value: Amount to increment by
        """
        self.count += value
        self.last_updated = time.time()
    
    def reset(self) -> None:
        """Reset the counter to zero"""
        self.count = 0
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary for serialization
        
        Returns:
            Dict: Metric as dictionary
        """
        result = super().to_dict()
        result["count"] = self.count
        return result


class GaugeMetric(Metric):
    """
    A metric that records a value that can go up and down.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a gauge metric
        
        Args:
            name: Metric name
            description: Human-readable description
        """
        super().__init__(name, description)
        self.value = 0.0
    
    def set(self, value: float) -> None:
        """
        Set the gauge value
        
        Args:
            value: New value to set
        """
        self.value = value
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary for serialization
        
        Returns:
            Dict: Metric as dictionary
        """
        result = super().to_dict()
        result["value"] = self.value
        return result


class HistogramMetric(Metric):
    """
    A metric that samples observations and counts them in configurable buckets.
    """
    
    def __init__(self, name: str, description: str, buckets: Optional[List[float]] = None):
        """
        Initialize a histogram metric
        
        Args:
            name: Metric name
            description: Human-readable description
            buckets: List of bucket boundaries (upper bounds)
        """
        super().__init__(name, description)
        self.buckets = buckets or [1, 5, 10, 50, 100, 500, 1000]
        self.observations = []
        self.count = 0
        self.sum = 0.0
    
    def observe(self, value: float) -> None:
        """
        Record an observation
        
        Args:
            value: Observed value
        """
        self.observations.append(value)
        self.count += 1
        self.sum += value
        self.last_updated = time.time()
        
        # Limit the number of stored observations to prevent memory issues
        if len(self.observations) > 1000:
            self.observations = self.observations[-1000:]
    
    def get_buckets(self) -> Dict[str, int]:
        """
        Get histogram buckets
        
        Returns:
            Dict: Bucket counts
        """
        result = {}
        for bucket in self.buckets:
            result[str(bucket)] = sum(1 for obs in self.observations if obs <= bucket)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary for serialization
        
        Returns:
            Dict: Metric as dictionary
        """
        result = super().to_dict()
        result["count"] = self.count
        result["sum"] = self.sum
        result["buckets"] = self.get_buckets()
        
        # Calculate some basic statistics
        if self.observations:
            result["avg"] = self.sum / self.count
            result["min"] = min(self.observations)
            result["max"] = max(self.observations)
        
        return result


class TelemetryManager:
    """
    Central manager for system telemetry and performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the telemetry manager
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.enabled = config.get("telemetry", {}).get("enabled", True)
        self.metrics: Dict[str, Metric] = {}
        self.collectors: List[Callable] = []
        self.running = False
        self.collection_thread = None
        self.collection_interval = config.get("telemetry", {}).get("collection_interval", 60)
        self.export_path = config.get("telemetry", {}).get("export_path", "data/telemetry")
        
        # Create metrics directory if it doesn't exist
        if self.enabled:
            os.makedirs(self.export_path, exist_ok=True)
    
    def register_metric(self, metric: Metric) -> None:
        """
        Register a metric to be tracked
        
        Args:
            metric: The metric to register
        """
        self.metrics[metric.name] = metric
        logger.debug(f"Registered metric: {metric.name}")
    
    def register_collector(self, collector: Callable) -> None:
        """
        Register a collector function that will be called to update metrics
        
        Args:
            collector: A function that collects metrics
        """
        self.collectors.append(collector)
        logger.debug(f"Registered collector function: {collector.__name__}")
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get a metric by name
        
        Args:
            name: Metric name
            
        Returns:
            Optional[Metric]: The metric if found, None otherwise
        """
        return self.metrics.get(name)
    
    def start_collection(self) -> None:
        """
        Start the metrics collection thread
        """
        if not self.enabled:
            logger.info("Telemetry is disabled. Not starting collection.")
            return
        
        if self.running:
            logger.warning("Telemetry collection is already running.")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        logger.info("Started telemetry collection thread")
    
    def stop_collection(self) -> None:
        """
        Stop the metrics collection thread
        """
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
            logger.info("Stopped telemetry collection thread")
    
    def _collection_loop(self) -> None:
        """Background thread for periodic metrics collection"""
        while self.running:
            try:
                # Call all registered collectors
                for collector in self.collectors:
                    try:
                        collector()
                    except Exception as e:
                        logger.error(f"Error in metric collector {collector.__name__}: {e}")
                
                # Export metrics
                self._export_metrics()
                
                # Wait for next collection interval
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in telemetry collection loop: {e}")
                time.sleep(5)  # Wait a bit before retrying
    
    def _export_metrics(self) -> None:
        """Export current metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.export_path, f"metrics_{timestamp}.json")
        
        try:
            # Convert all metrics to dictionaries
            metrics_data = {
                name: metric.to_dict() for name, metric in self.metrics.items()
            }
            
            # Add metadata
            export_data = {
                "timestamp": timestamp,
                "metrics": metrics_data
            }
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.debug(f"Exported metrics to {filename}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all metrics as dictionaries
        
        Returns:
            Dict: All metrics
        """
        return {name: metric.to_dict() for name, metric in self.metrics.items()}


# Performance tracking utilities
class PerformanceTracker:
    """
    Utility for tracking execution time of operations.
    """
    
    def __init__(self, name: str, telemetry_manager: Optional[TelemetryManager] = None):
        """
        Initialize a performance tracker
        
        Args:
            name: Name of the component being tracked
            telemetry_manager: Optional telemetry manager to register metrics with
        """
        self.name = name
        self.telemetry_manager = telemetry_manager
        self.timing_histogram = None
        
        # Register with telemetry if provided
        if telemetry_manager:
            metric_name = f"{name}_execution_time"
            self.timing_histogram = HistogramMetric(
                metric_name, 
                f"Execution time for {name} operations (ms)"
            )
            telemetry_manager.register_metric(self.timing_histogram)
    
    def measure_execution_time(self, operation_name: str) -> Callable:
        """
        Decorator to measure execution time of a function
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Callable: Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Record the time
                logger.debug(f"{self.name} - {operation_name}: {execution_time:.2f} ms")
                
                # Add to telemetry if available
                if self.timing_histogram:
                    self.timing_histogram.observe(execution_time)
                
                return result
            return wrapper
        return decorator
    
    def timed_section(self, operation_name: str):
        """
        Context manager for timing a section of code
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Context manager for timing
        """
        class TimedContext:
            def __init__(self, tracker, op_name):
                self.tracker = tracker
                self.op_name = op_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                execution_time = (time.time() - self.start_time) * 1000  # Convert to ms
                
                # Record the time
                logger.debug(f"{self.tracker.name} - {self.op_name}: {execution_time:.2f} ms")
                
                # Add to telemetry if available
                if self.tracker.timing_histogram:
                    self.tracker.timing_histogram.observe(execution_time)
        
        return TimedContext(self, operation_name)