"""
Performance Metrics System

This module implements metrics tracking and performance evaluation
for the AI agent to measure progress and compare against human performance.
"""

import logging
import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Setup module-level logger
logger = logging.getLogger("wow_ai.learning.performance_metrics")

@dataclass
class PerformanceRecord:
    """Single performance measurement record"""
    metric_name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


class MetricTracker:
    """Tracks the history of a specific metric"""
    
    def __init__(self, name: str, description: str = "", unit: str = ""):
        """
        Initialize a metric tracker
        
        Args:
            name: Metric name
            description: Metric description
            unit: Measurement unit
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.records: List[PerformanceRecord] = []
        
    def add_record(self, value: float, context: Dict[str, Any] = None) -> None:
        """
        Add a new record for this metric
        
        Args:
            value: Metric value
            context: Additional contextual information
        """
        if context is None:
            context = {}
            
        record = PerformanceRecord(
            metric_name=self.name,
            value=value,
            context=context
        )
        self.records.append(record)
        
    def get_latest_value(self) -> Optional[float]:
        """
        Get the most recent value for this metric
        
        Returns:
            Latest metric value or None if no records exist
        """
        if not self.records:
            return None
        
        return self.records[-1].value
    
    def get_average(self, window: int = None) -> Optional[float]:
        """
        Get the average value over a window of records
        
        Args:
            window: Number of most recent records to average, or None for all records
            
        Returns:
            Average value or None if no records exist
        """
        if not self.records:
            return None
        
        if window is None or window >= len(self.records):
            values = [record.value for record in self.records]
        else:
            values = [record.value for record in self.records[-window:]]
        
        return sum(values) / len(values)
    
    def get_trend(self, window: int = 10) -> Optional[float]:
        """
        Calculate the trend (slope) of recent values
        
        Args:
            window: Number of most recent records to analyze
            
        Returns:
            Trend value (positive = improving, negative = declining) or None if insufficient data
        """
        if len(self.records) < 2:
            return None
        
        records = self.records[-min(window, len(self.records)):]
        values = np.array([record.value for record in records])
        
        # Simple linear regression to find trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return slope
    
    def get_percentile(self, percentile: float = 95) -> Optional[float]:
        """
        Calculate a percentile value of all records
        
        Args:
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value or None if no records exist
        """
        if not self.records:
            return None
        
        values = [record.value for record in self.records]
        return np.percentile(values, percentile)


class PerformanceMetricsManager:
    """Manager for tracking and analyzing performance metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the performance metrics manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics: Dict[str, MetricTracker] = {}
        self.human_benchmarks: Dict[str, float] = {}
        self.session_start_time = time.time()
        
        # Initialize standard metrics
        self._init_standard_metrics()
        
        # Load existing metrics and benchmarks if available
        self._load_metrics()
        self._load_human_benchmarks()
        
    def _init_standard_metrics(self) -> None:
        """Initialize standard performance metrics"""
        standard_metrics = [
            # Combat metrics
            ("combat_dps", "Damage per second in combat", "DPS"),
            ("combat_survival_time", "Average survival time in combat", "seconds"),
            ("combat_kill_time", "Average time to kill enemies", "seconds"),
            ("combat_resource_efficiency", "Efficient use of class resources", "percent"),
            
            # Leveling metrics
            ("leveling_speed", "Time required per level", "minutes"),
            ("leveling_quests_per_hour", "Quests completed per hour", "quests/hour"),
            ("leveling_xp_per_hour", "Experience gained per hour", "XP/hour"),
            
            # Quest metrics
            ("quest_completion_time", "Average time to complete quests", "minutes"),
            ("quest_efficiency", "Quest objectives completed per minute", "objectives/minute"),
            
            # Movement metrics
            ("movement_efficiency", "Distance traveled vs optimal path", "percent"),
            ("navigation_time", "Time to navigate to destinations", "seconds"),
            
            # Decision metrics
            ("decision_time", "Time to make decisions", "ms"),
            ("action_success_rate", "Rate of successful actions", "percent"),
            
            # Resource metrics
            ("gold_per_hour", "Gold earned per hour", "gold/hour"),
            ("inventory_efficiency", "Inventory management efficiency", "percent")
        ]
        
        for name, description, unit in standard_metrics:
            self.add_metric(name, description, unit)
            
    def add_metric(self, name: str, description: str = "", unit: str = "") -> MetricTracker:
        """
        Add a new metric to track
        
        Args:
            name: Metric name
            description: Metric description
            unit: Measurement unit
            
        Returns:
            The created metric tracker
        """
        if name not in self.metrics:
            self.metrics[name] = MetricTracker(name, description, unit)
        return self.metrics[name]
    
    def record_metric(self, name: str, value: float, context: Dict[str, Any] = None) -> None:
        """
        Record a metric value
        
        Args:
            name: Metric name
            value: Metric value
            context: Additional contextual information
        """
        if name not in self.metrics:
            self.add_metric(name)
            
        self.metrics[name].add_record(value, context)
        
    def get_metric(self, name: str) -> Optional[MetricTracker]:
        """
        Get a metric tracker by name
        
        Args:
            name: Metric name
            
        Returns:
            The metric tracker or None if not found
        """
        return self.metrics.get(name)
    
    def set_human_benchmark(self, metric_name: str, value: float) -> None:
        """
        Set a human performance benchmark for comparison
        
        Args:
            metric_name: Metric name
            value: Benchmark value
        """
        self.human_benchmarks[metric_name] = value
        
    def compare_to_human(self, metric_name: str) -> Optional[float]:
        """
        Compare current performance to human benchmark
        
        Args:
            metric_name: Metric name
            
        Returns:
            Ratio of AI performance to human benchmark (>1 means AI is better),
            or None if metric or benchmark doesn't exist
        """
        if metric_name not in self.metrics or metric_name not in self.human_benchmarks:
            return None
        
        current_value = self.metrics[metric_name].get_latest_value()
        if current_value is None:
            return None
        
        benchmark = self.human_benchmarks[metric_name]
        
        # For some metrics, higher is better, for others lower is better
        # Determine this based on metric name
        if any(term in metric_name for term in ["time", "distance"]):
            # For time/distance metrics, lower is better
            return benchmark / current_value if current_value > 0 else None
        else:
            # For most other metrics, higher is better
            return current_value / benchmark if benchmark > 0 else None
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a performance summary across all metrics
        
        Returns:
            Dictionary with performance summary
        """
        summary = {
            "session_duration": time.time() - self.session_start_time,
            "metrics": {},
            "human_comparisons": {},
            "trends": {},
            "overall_rating": 0.0
        }
        
        # Compile metrics data
        for name, tracker in self.metrics.items():
            latest = tracker.get_latest_value()
            avg = tracker.get_average(window=10)
            trend = tracker.get_trend()
            
            if latest is not None:
                summary["metrics"][name] = {
                    "latest": latest,
                    "average": avg,
                    "trend": trend,
                    "unit": tracker.unit
                }
            
            # Add human comparison if available
            comparison = self.compare_to_human(name)
            if comparison is not None:
                summary["human_comparisons"][name] = comparison
        
        # Calculate overall performance rating
        # This is a weighted average of all human comparisons
        if summary["human_comparisons"]:
            comparisons = list(summary["human_comparisons"].values())
            summary["overall_rating"] = sum(comparisons) / len(comparisons)
        
        return summary
    
    def _load_metrics(self) -> None:
        """Load metrics history from disk"""
        metrics_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        metrics_path = os.path.join(metrics_dir, "performance_metrics.json")
        
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    
                for metric_name, metric_data in data.items():
                    if metric_name not in self.metrics:
                        self.add_metric(
                            metric_name,
                            metric_data.get("description", ""),
                            metric_data.get("unit", "")
                        )
                    
                    # Load records
                    metric = self.metrics[metric_name]
                    for record in metric_data.get("records", []):
                        metric.add_record(
                            record["value"],
                            record.get("context", {})
                        )
                        
                logger.info("Loaded performance metrics history")
            except Exception as e:
                logger.error(f"Failed to load metrics: {e}")
    
    def _load_human_benchmarks(self) -> None:
        """Load human performance benchmarks"""
        metrics_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        benchmarks_path = os.path.join(metrics_dir, "human_benchmarks.json")
        
        if os.path.exists(benchmarks_path):
            try:
                with open(benchmarks_path, 'r') as f:
                    self.human_benchmarks = json.load(f)
                    
                logger.info("Loaded human performance benchmarks")
            except Exception as e:
                logger.error(f"Failed to load human benchmarks: {e}")
    
    def save_metrics(self) -> None:
        """Save metrics history to disk"""
        metrics_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "performance_metrics.json")
        
        # Convert metrics to serializable format
        data = {}
        for name, tracker in self.metrics.items():
            data[name] = {
                "description": tracker.description,
                "unit": tracker.unit,
                "records": [
                    {
                        "value": record.value,
                        "timestamp": record.timestamp,
                        "context": record.context
                    }
                    for record in tracker.records
                ]
            }
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("Saved performance metrics history")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def save_human_benchmarks(self) -> None:
        """Save human benchmarks to disk"""
        metrics_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "learning"
        )
        
        os.makedirs(metrics_dir, exist_ok=True)
        benchmarks_path = os.path.join(metrics_dir, "human_benchmarks.json")
        
        try:
            with open(benchmarks_path, 'w') as f:
                json.dump(self.human_benchmarks, f, indent=2)
                
            logger.info("Saved human performance benchmarks")
        except Exception as e:
            logger.error(f"Failed to save human benchmarks: {e}")
    
    def generate_performance_report(self, output_path: str = None) -> str:
        """
        Generate a detailed performance report
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report text
        """
        summary = self.get_performance_summary()
        
        # Build report text
        report = [
            "# AI Agent Performance Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session duration: {timedelta(seconds=int(summary['session_duration']))}",
            "",
            "## Performance Summary",
            f"Overall rating: {summary['overall_rating']:.2f} x Human Performance" if summary['overall_rating'] else "Overall rating: Not available",
            "",
            "## Key Metrics"
        ]
        
        # Add metrics details
        for name, data in summary["metrics"].items():
            human_comparison = summary.get("human_comparisons", {}).get(name)
            
            report.append(f"### {name} ({data['unit']})")
            report.append(f"- Current: {data['latest']:.2f}")
            report.append(f"- Average (last 10): {data['average']:.2f}")
            
            if data['trend'] is not None:
                trend_direction = "improving" if data['trend'] > 0 else "declining"
                report.append(f"- Trend: {data['trend']:.4f} ({trend_direction})")
            
            if human_comparison is not None:
                report.append(f"- vs Human: {human_comparison:.2f}x" + 
                             (" (better)" if human_comparison > 1 else " (worse)"))
            
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save report if output path provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(report_text)
            except Exception as e:
                logger.error(f"Failed to save performance report: {e}")
        
        return report_text