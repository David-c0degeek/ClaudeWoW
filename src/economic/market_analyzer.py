"""
Market analysis system for auction house data processing and price prediction.
"""
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from dataclasses import dataclass

from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class ItemPrice:
    """Represents pricing data for an item."""
    item_id: int
    name: str
    min_price: int
    market_price: int
    timestamp: datetime
    quantity: int = 1
    
    @property
    def gold_formatted(self) -> str:
        """Return the price formatted as gold, silver, copper."""
        gold = self.min_price // 10000
        silver = (self.min_price % 10000) // 100
        copper = self.min_price % 100
        return f"{gold}g {silver}s {copper}c"

class MarketAnalyzer:
    """
    Analyzes auction house data to identify market trends and opportunities.
    
    This class handles:
    - Auction house data collection and parsing
    - Price trend detection and prediction
    - Item value assessment
    - Arbitrage opportunity identification
    """
    
    def __init__(self, config: Config):
        """
        Initialize the MarketAnalyzer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.price_history: Dict[int, List[ItemPrice]] = {}
        self.price_data_file = self.config.get("paths.economic.price_data", "data/economic/price_data.json")
        self.load_price_data()
        
    def load_price_data(self) -> None:
        """Load existing price data from disk."""
        try:
            with open(self.price_data_file, 'r') as f:
                data = json.load(f)
                
            for item_id, prices in data.items():
                item_id = int(item_id)
                self.price_history[item_id] = []
                for p in prices:
                    price = ItemPrice(
                        item_id=item_id,
                        name=p["name"],
                        min_price=p["min_price"],
                        market_price=p["market_price"],
                        timestamp=datetime.fromisoformat(p["timestamp"]),
                        quantity=p.get("quantity", 1)
                    )
                    self.price_history[item_id].append(price)
                    
            logger.info(f"Loaded price data for {len(self.price_history)} items")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No previous price data found or file is corrupted")
            self.price_history = {}
            
    def save_price_data(self) -> None:
        """Save price data to disk."""
        data = {}
        for item_id, prices in self.price_history.items():
            data[str(item_id)] = []
            for price in prices:
                data[str(item_id)].append({
                    "name": price.name,
                    "min_price": price.min_price,
                    "market_price": price.market_price,
                    "timestamp": price.timestamp.isoformat(),
                    "quantity": price.quantity
                })
                
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.price_data_file), exist_ok=True)
            
            with open(self.price_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved price data for {len(self.price_history)} items")
        except Exception as e:
            logger.error(f"Failed to save price data: {e}")
            
    def update_prices(self, current_auction_data: List[Dict]) -> None:
        """
        Update price history with new auction house data.
        
        Args:
            current_auction_data: List of auction items with pricing
        """
        timestamp = datetime.now()
        
        for item in current_auction_data:
            item_id = item["id"]
            price = ItemPrice(
                item_id=item_id,
                name=item["name"],
                min_price=item["buyout_price"],
                market_price=item.get("market_price", item["buyout_price"]),
                timestamp=timestamp,
                quantity=item.get("quantity", 1)
            )
            
            if item_id not in self.price_history:
                self.price_history[item_id] = []
                
            self.price_history[item_id].append(price)
            
        # Prune old data (older than 30 days)
        self._prune_old_data()
        
        # Save updated data
        self.save_price_data()
        
    def _prune_old_data(self, days: int = 30) -> None:
        """
        Remove price data older than specified days.
        
        Args:
            days: Number of days to keep data for
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        for item_id in self.price_history:
            self.price_history[item_id] = [
                p for p in self.price_history[item_id] 
                if p.timestamp >= cutoff
            ]
            
    def get_item_trend(self, item_id: int) -> Dict:
        """
        Analyze price trend for a specific item.
        
        Args:
            item_id: ID of the item to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if item_id not in self.price_history or not self.price_history[item_id]:
            return {"trend": "unknown", "confidence": 0, "data": []}
            
        prices = self.price_history[item_id]
        
        # Sort by timestamp
        prices.sort(key=lambda x: x.timestamp)
        
        # Get price points for analysis
        price_points = [p.min_price for p in prices]
        
        # Calculate trend
        if len(price_points) < 2:
            return {
                "trend": "stable",
                "confidence": 0,
                "current_price": prices[-1].min_price,
                "name": prices[-1].name,
                "data": price_points
            }
            
        # Linear regression
        x = np.arange(len(price_points))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, price_points)
        
        # Determine trend
        trend = "stable"
        if slope > 0:
            if slope / price_points[0] > 0.05:  # 5% increase
                trend = "rising"
            else:
                trend = "slight_rise"
        elif slope < 0:
            if abs(slope) / price_points[0] > 0.05:  # 5% decrease
                trend = "falling"
            else:
                trend = "slight_fall"
                
        # Calculate confidence
        confidence = min(abs(r_value), 1.0)
        
        return {
            "trend": trend,
            "confidence": confidence,
            "slope": slope,
            "current_price": prices[-1].min_price,
            "name": prices[-1].name,
            "data": price_points
        }
        
    def predict_price(self, item_id: int, days_ahead: int = 1) -> Optional[int]:
        """
        Predict price for a specific item in the future.
        
        Args:
            item_id: ID of the item to predict
            days_ahead: Number of days in the future to predict
            
        Returns:
            Predicted price or None if not enough data
        """
        if item_id not in self.price_history or len(self.price_history[item_id]) < 5:
            return None
            
        prices = self.price_history[item_id]
        
        # Sort by timestamp
        prices.sort(key=lambda x: x.timestamp)
        
        # Get price points for analysis
        price_points = [p.min_price for p in prices]
        
        # Linear regression
        x = np.arange(len(price_points))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, price_points)
        
        # Predict future price
        future_x = len(price_points) + days_ahead
        predicted_price = max(round(intercept + slope * future_x), 1)  # Ensure minimum price of 1
        
        return predicted_price
        
    def find_arbitrage_opportunities(self, min_roi: float = 0.15) -> List[Dict]:
        """
        Find potential arbitrage opportunities across servers or markets.
        
        Args:
            min_roi: Minimum return on investment required (15% by default)
            
        Returns:
            List of arbitrage opportunities
        """
        # In a real system, this would compare prices across different servers
        # For now, we'll simulate by comparing current prices to predicted prices
        opportunities = []
        
        for item_id, prices in self.price_history.items():
            if len(prices) < 5:
                continue
                
            current_price = sorted(prices, key=lambda x: x.timestamp)[-1].min_price
            predicted_price = self.predict_price(item_id, days_ahead=3)
            
            if not predicted_price:
                continue
                
            roi = (predicted_price - current_price) / current_price
            
            if roi >= min_roi:
                item_name = prices[0].name
                opportunities.append({
                    "item_id": item_id,
                    "name": item_name,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "roi": roi,
                    "confidence": self.get_item_trend(item_id)["confidence"]
                })
                
        # Sort by ROI
        opportunities.sort(key=lambda x: x["roi"], reverse=True)
        
        return opportunities
        
    def get_most_profitable_items(self, n: int = 10) -> List[Dict]:
        """
        Get the most profitable items to sell.
        
        Args:
            n: Number of items to return
            
        Returns:
            List of most profitable items
        """
        items = []
        
        for item_id, prices in self.price_history.items():
            if not prices:
                continue
                
            latest_price = sorted(prices, key=lambda x: x.timestamp)[-1]
            
            # Calculate volume and volatility
            if len(prices) >= 3:
                volumes = [p.quantity for p in prices]
                price_values = [p.min_price for p in prices]
                volatility = np.std(price_values) / np.mean(price_values) if np.mean(price_values) > 0 else 0
                avg_volume = sum(volumes) / len(volumes)
            else:
                volatility = 0
                avg_volume = latest_price.quantity
                
            items.append({
                "item_id": item_id,
                "name": latest_price.name,
                "price": latest_price.min_price,
                "volume": avg_volume,
                "volatility": volatility,
                "score": latest_price.min_price * avg_volume * (1 + volatility)  # Profit potential score
            })
            
        # Sort by score
        items.sort(key=lambda x: x["score"], reverse=True)
        
        return items[:n]