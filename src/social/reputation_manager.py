# src/social/reputation_manager.py

import logging
import time
import json
import os
import random
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

class ReputationManager:
    """
    Manages player reputation and relationships across different social contexts
    
    This system tracks relationships with:
    - Individual players
    - Guilds
    - Trading partners
    - Server communities
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ReputationManager
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.social.reputation_manager")
        self.config = config
        
        # Reputation storage paths
        self.data_path = config.get("data_path", "data")
        self.reputation_path = os.path.join(self.data_path, "social", "reputation")
        
        # Make sure directories exist
        os.makedirs(self.reputation_path, exist_ok=True)
        
        # Relationship thresholds
        self.relationship_thresholds = {
            "hated": -50,
            "disliked": -25,
            "neutral": 0,
            "friendly": 25,
            "trusted": 50,
            "honored": 75
        }
        
        # Relationship decay rates - points lost per day without interaction
        self.daily_decay_rates = {
            "hated": 1,       # Hatred decreases slightly over time
            "disliked": 2,     # Dislike decreases moderately over time
            "neutral": 0,      # Neutral relationships don't decay
            "friendly": -1,    # Friendly decreases if not maintained
            "trusted": -2,     # Trust decreases faster if not maintained
            "honored": -3      # Honor status decreases fastest if not maintained
        }
        
        # Load reputation data
        self.players = self._load_reputation_data("players")
        self.guilds = self._load_reputation_data("guilds")
        self.trade_partners = self._load_reputation_data("trade_partners")
        self.communities = self._load_reputation_data("communities")
        
        # Initialize trusted player heuristics
        self.trust_heuristics = {
            "mutual_group_experience": 5,      # Points for successful group activities
            "fair_trade_completed": 3,         # Points for successful trades
            "chat_reciprocation": 1,           # Points for positive chat interactions
            "received_help": 8,                # Points for receiving help
            "provided_help": 5,                # Points for providing help
            "scam_attempt": -50,               # Major penalty for scamming
            "ninja_looting": -30,              # Major penalty for ninja looting
            "verbal_abuse": -15,               # Penalty for harassment/abuse
            "group_abandonment": -10,          # Penalty for leaving group mid-activity
            "declined_reasonable_request": -3,  # Small penalty for declining reasonable help
            "unexpected_gift": 10              # Points for receiving unsolicited gift/help
        }
        
        # Community contribution tracking
        self.community_contribution_types = {
            "answering_questions": 2,          # Points for answering questions in chat
            "guild_bank_donation": 5,          # Points for donating to guild bank
            "newbie_assistance": 5,            # Points for helping new players
            "group_formation": 3,              # Points for forming groups
            "market_price_stabilization": 3,   # Points for fair market pricing
            "carrying_lowbies": 4              # Points for helping lower level players
        }
        
        self.logger.info("ReputationManager initialized")
    
    def _load_reputation_data(self, category: str) -> Dict:
        """
        Load reputation data from storage
        
        Args:
            category: Category of reputation data (players, guilds, etc.)
            
        Returns:
            Dict: Reputation data
        """
        filepath = os.path.join(self.reputation_path, f"{category}.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"Could not parse {category} reputation data")
                return {}
        else:
            # Create empty reputation file
            empty_data = {}
            try:
                with open(filepath, 'w') as f:
                    json.dump(empty_data, f)
            except Exception as e:
                self.logger.error(f"Failed to create empty {category} reputation file: {e}")
            
            return empty_data
    
    def _save_reputation_data(self, category: str, data: Dict):
        """
        Save reputation data to storage
        
        Args:
            category: Category of reputation data
            data: Data to save
        """
        filepath = os.path.join(self.reputation_path, f"{category}.json")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save {category} reputation data: {e}")
    
    def update_player_reputation(self, player_name: str, event_type: str, context: Optional[Dict] = None) -> int:
        """
        Update a player's reputation based on an interaction event
        
        Args:
            player_name: Name of the player
            event_type: Type of interaction (from trust_heuristics)
            context: Additional context about the interaction
            
        Returns:
            int: New reputation score
        """
        if not player_name or player_name == "":
            return 0
            
        # Initialize player record if needed
        if player_name not in self.players:
            self.players[player_name] = {
                "reputation": 0,
                "relationship": "neutral",
                "first_seen": time.time(),
                "last_seen": time.time(),
                "interactions": 0,
                "interaction_history": [],
                "notes": []
            }
        
        # Update last seen time
        self.players[player_name]["last_seen"] = time.time()
        self.players[player_name]["interactions"] += 1
        
        # Calculate reputation change
        rep_change = 0
        if event_type in self.trust_heuristics:
            rep_change = self.trust_heuristics[event_type]
        
        # Apply change with context adjustments
        if context:
            # Adjust based on magnitude
            magnitude = context.get("magnitude", 1.0)
            rep_change = int(rep_change * magnitude)
            
            # Record notes if provided
            if "note" in context:
                self.players[player_name]["notes"].append({
                    "timestamp": time.time(),
                    "note": context["note"]
                })
        
        # Update reputation
        self.players[player_name]["reputation"] += rep_change
        
        # Record interaction
        self.players[player_name]["interaction_history"].append({
            "timestamp": time.time(),
            "event_type": event_type,
            "reputation_change": rep_change,
            "context": context or {}
        })
        
        # Trim history if needed
        max_history = self.config.get("max_interaction_history", 50)
        if len(self.players[player_name]["interaction_history"]) > max_history:
            self.players[player_name]["interaction_history"] = self.players[player_name]["interaction_history"][-max_history:]
        
        # Update relationship level
        self._update_relationship_level(player_name)
        
        # Save changes
        self._save_reputation_data("players", self.players)
        
        return self.players[player_name]["reputation"]
    
    def _update_relationship_level(self, player_name: str):
        """
        Update a player's relationship level based on their reputation score
        
        Args:
            player_name: Name of the player
        """
        rep = self.players[player_name]["reputation"]
        
        # Find the appropriate relationship level
        current_level = "neutral"
        for level, threshold in sorted(self.relationship_thresholds.items(), key=lambda x: x[1]):
            if rep >= threshold:
                current_level = level
        
        # Update relationship level
        self.players[player_name]["relationship"] = current_level
    
    def update_guild_reputation(self, guild_name: str, event_type: str, context: Optional[Dict] = None) -> int:
        """
        Update a guild's reputation based on an interaction event
        
        Args:
            guild_name: Name of the guild
            event_type: Type of interaction
            context: Additional context about the interaction
            
        Returns:
            int: New reputation score
        """
        if not guild_name or guild_name == "":
            return 0
            
        # Initialize guild record if needed
        if guild_name not in self.guilds:
            self.guilds[guild_name] = {
                "reputation": 0,
                "relationship": "neutral",
                "first_seen": time.time(),
                "last_seen": time.time(),
                "interactions": 0,
                "known_members": [],
                "interaction_history": [],
                "guild_type": context.get("guild_type", "unknown") if context else "unknown",
                "notes": []
            }
        
        # Update last seen time
        self.guilds[guild_name]["last_seen"] = time.time()
        self.guilds[guild_name]["interactions"] += 1
        
        # Add member if provided
        if context and "member_name" in context:
            member = context["member_name"]
            if member not in self.guilds[guild_name]["known_members"]:
                self.guilds[guild_name]["known_members"].append(member)
        
        # Calculate reputation change
        rep_change = 0
        if event_type in self.trust_heuristics:
            rep_change = self.trust_heuristics[event_type]
        
        # Apply change with context adjustments
        if context:
            # Adjust based on magnitude
            magnitude = context.get("magnitude", 1.0)
            rep_change = int(rep_change * magnitude)
            
            # Record notes if provided
            if "note" in context:
                self.guilds[guild_name]["notes"].append({
                    "timestamp": time.time(),
                    "note": context["note"]
                })
                
            # Update guild type if provided
            if "guild_type" in context:
                self.guilds[guild_name]["guild_type"] = context["guild_type"]
        
        # Update reputation
        self.guilds[guild_name]["reputation"] += rep_change
        
        # Record interaction
        self.guilds[guild_name]["interaction_history"].append({
            "timestamp": time.time(),
            "event_type": event_type,
            "reputation_change": rep_change,
            "context": context or {}
        })
        
        # Update relationship level
        self._update_guild_relationship_level(guild_name)
        
        # Save changes
        self._save_reputation_data("guilds", self.guilds)
        
        return self.guilds[guild_name]["reputation"]
    
    def _update_guild_relationship_level(self, guild_name: str):
        """
        Update a guild's relationship level based on their reputation score
        
        Args:
            guild_name: Name of the guild
        """
        rep = self.guilds[guild_name]["reputation"]
        
        # Find the appropriate relationship level
        current_level = "neutral"
        for level, threshold in sorted(self.relationship_thresholds.items(), key=lambda x: x[1]):
            if rep >= threshold:
                current_level = level
        
        # Update relationship level
        self.guilds[guild_name]["relationship"] = current_level
    
    def update_trade_partner_reputation(self, player_name: str, trade_type: str, trade_value: int, 
                                       was_fair: bool, context: Optional[Dict] = None) -> int:
        """
        Update a trading partner's reputation based on a trade
        
        Args:
            player_name: Name of the player
            trade_type: Type of trade (buy, sell)
            trade_value: Value of the trade in gold
            was_fair: Whether the trade was fair
            context: Additional context about the interaction
            
        Returns:
            int: New reputation score
        """
        if not player_name or player_name == "":
            return 0
            
        # Initialize trade partner record if needed
        if player_name not in self.trade_partners:
            self.trade_partners[player_name] = {
                "reputation": 0,
                "relationship": "neutral",
                "first_trade": time.time(),
                "last_trade": time.time(),
                "trade_count": 0,
                "total_value": 0,
                "trades": [],
                "honest_trades": 0,
                "unfair_trades": 0
            }
        
        # Update trade stats
        self.trade_partners[player_name]["last_trade"] = time.time()
        self.trade_partners[player_name]["trade_count"] += 1
        self.trade_partners[player_name]["total_value"] += trade_value
        
        if was_fair:
            self.trade_partners[player_name]["honest_trades"] += 1
            event_type = "fair_trade_completed"
        else:
            self.trade_partners[player_name]["unfair_trades"] += 1
            event_type = "scam_attempt"
        
        # Record trade
        trade_record = {
            "timestamp": time.time(),
            "trade_type": trade_type,
            "value": trade_value,
            "was_fair": was_fair
        }
        
        if context:
            trade_record.update(context)
        
        self.trade_partners[player_name]["trades"].append(trade_record)
        
        # Trim history if needed
        max_trades = self.config.get("max_trade_history", 50)
        if len(self.trade_partners[player_name]["trades"]) > max_trades:
            self.trade_partners[player_name]["trades"] = self.trade_partners[player_name]["trades"][-max_trades:]
        
        # Update player's general reputation as well
        fairness_magnitude = 1.0
        if trade_value > 50:
            fairness_magnitude = 2.0  # Higher value trades impact reputation more
        elif trade_value > 200:
            fairness_magnitude = 3.0
            
        context = context or {}
        context["magnitude"] = fairness_magnitude
        context["trade_value"] = trade_value
        
        rep_score = self.update_player_reputation(player_name, event_type, context)
        
        # Save changes
        self._save_reputation_data("trade_partners", self.trade_partners)
        
        return rep_score
    
    def update_community_reputation(self, action_type: str, context: Optional[Dict] = None) -> Dict:
        """
        Update reputation across a community based on a contribution
        
        Args:
            action_type: Type of community contribution
            context: Additional context including target community
            
        Returns:
            Dict: Updated community reputation info
        """
        if not context or "community" not in context:
            return {}
            
        community = context["community"]
        
        # Initialize community record if needed
        if community not in self.communities:
            self.communities[community] = {
                "standing": 0,
                "first_interaction": time.time(),
                "last_interaction": time.time(),
                "interaction_count": 0,
                "contribution_history": [],
                "key_members": []
            }
        
        # Update interaction stats
        self.communities[community]["last_interaction"] = time.time()
        self.communities[community]["interaction_count"] += 1
        
        # Calculate reputation change
        rep_change = 0
        if action_type in self.community_contribution_types:
            rep_change = self.community_contribution_types[action_type]
        
        # Apply context modifications
        if context:
            magnitude = context.get("magnitude", 1.0)
            rep_change = int(rep_change * magnitude)
            
            # Add key member if provided
            if "member_name" in context:
                member = context["member_name"]
                if member not in self.communities[community]["key_members"]:
                    self.communities[community]["key_members"].append(member)
        
        # Update standing
        self.communities[community]["standing"] += rep_change
        
        # Record contribution
        self.communities[community]["contribution_history"].append({
            "timestamp": time.time(),
            "action_type": action_type,
            "standing_change": rep_change,
            "context": context
        })
        
        # Trim history if needed
        max_history = self.config.get("max_community_history", 50)
        if len(self.communities[community]["contribution_history"]) > max_history:
            self.communities[community]["contribution_history"] = self.communities[community]["contribution_history"][-max_history:]
        
        # Save changes
        self._save_reputation_data("communities", self.communities)
        
        return self.communities[community]
    
    def apply_reputation_decay(self):
        """
        Apply time-based decay to reputation scores
        
        This should be called periodically (e.g., daily) to simulate
        relationship decay over time.
        """
        current_time = time.time()
        day_seconds = 86400  # Seconds in a day
        
        # Apply decay to player relationships
        for player_name, data in self.players.items():
            days_since_interaction = (current_time - data["last_seen"]) / day_seconds
            
            # Only apply decay if it's been at least one day
            if days_since_interaction >= 1:
                relationship = data["relationship"]
                decay_rate = self.daily_decay_rates.get(relationship, 0)
                
                # Calculate total decay
                total_decay = int(decay_rate * days_since_interaction)
                
                if total_decay != 0:
                    data["reputation"] += total_decay  # Note: decay rates can be positive or negative
                    
                    # Add decay note to history
                    data["interaction_history"].append({
                        "timestamp": current_time,
                        "event_type": "time_decay",
                        "reputation_change": total_decay,
                        "context": {"days_since_interaction": days_since_interaction}
                    })
                    
                    # Update relationship level
                    self._update_relationship_level(player_name)
        
        # Apply similar decay to guilds
        for guild_name, data in self.guilds.items():
            days_since_interaction = (current_time - data["last_seen"]) / day_seconds
            
            if days_since_interaction >= 1:
                relationship = data["relationship"]
                decay_rate = self.daily_decay_rates.get(relationship, 0)
                
                total_decay = int(decay_rate * days_since_interaction)
                
                if total_decay != 0:
                    data["reputation"] += total_decay
                    
                    # Add decay note
                    data["interaction_history"].append({
                        "timestamp": current_time,
                        "event_type": "time_decay",
                        "reputation_change": total_decay,
                        "context": {"days_since_interaction": days_since_interaction}
                    })
                    
                    # Update relationship level
                    self._update_guild_relationship_level(guild_name)
        
        # Save all updated data
        self._save_reputation_data("players", self.players)
        self._save_reputation_data("guilds", self.guilds)
    
    def get_player_relation(self, player_name: str) -> Dict:
        """
        Get a player's relationship data
        
        Args:
            player_name: Name of the player
            
        Returns:
            Dict: Relationship data or default values if player unknown
        """
        if player_name in self.players:
            return self.players[player_name]
        else:
            # Return default values for unknown players
            return {
                "reputation": 0,
                "relationship": "neutral",
                "first_seen": time.time(),
                "last_seen": time.time(),
                "interactions": 0,
                "interaction_history": [],
                "notes": []
            }
    
    def get_guild_relation(self, guild_name: str) -> Dict:
        """
        Get a guild's relationship data
        
        Args:
            guild_name: Name of the guild
            
        Returns:
            Dict: Relationship data or default values if guild unknown
        """
        if guild_name in self.guilds:
            return self.guilds[guild_name]
        else:
            # Return default values for unknown guilds
            return {
                "reputation": 0,
                "relationship": "neutral",
                "first_seen": time.time(),
                "last_seen": time.time(),
                "interactions": 0,
                "known_members": [],
                "interaction_history": [],
                "guild_type": "unknown",
                "notes": []
            }
    
    def get_trade_relation(self, player_name: str) -> Dict:
        """
        Get a trading partner's relationship data
        
        Args:
            player_name: Name of the player
            
        Returns:
            Dict: Trading relationship data or default values if unknown
        """
        if player_name in self.trade_partners:
            return self.trade_partners[player_name]
        else:
            # Return default values for unknown trade partners
            return {
                "reputation": 0,
                "relationship": "neutral",
                "first_trade": time.time(),
                "last_trade": time.time(),
                "trade_count": 0,
                "total_value": 0,
                "trades": [],
                "honest_trades": 0,
                "unfair_trades": 0
            }
    
    def get_top_relations(self, relation_type: str = "players", count: int = 5, sort_key: str = "reputation") -> List[Dict]:
        """
        Get top relationships sorted by a specific key
        
        Args:
            relation_type: Type of relationships to query (players, guilds, trade_partners)
            count: Number of results to return
            sort_key: Key to sort by (reputation, interactions, etc.)
            
        Returns:
            List[Dict]: Top relationships with names and data
        """
        data_source = getattr(self, relation_type, {})
        if not data_source:
            return []
            
        # Sort the relationships
        sorted_relations = []
        for name, data in data_source.items():
            if sort_key in data:
                sorted_relations.append({
                    "name": name,
                    "data": data
                })
        
        sorted_relations.sort(key=lambda x: x["data"].get(sort_key, 0), reverse=True)
        
        # Return top N results
        return sorted_relations[:count]
    
    def generate_relationship_advice(self, player_name: str = None, guild_name: str = None) -> Dict:
        """
        Generate advice for improving a specific relationship
        
        Args:
            player_name: Optional player name to get advice for
            guild_name: Optional guild name to get advice for
            
        Returns:
            Dict: Relationship advice
        """
        advice = {
            "target": player_name or guild_name,
            "current_status": "neutral",
            "suggestions": []
        }
        
        if player_name and player_name in self.players:
            data = self.players[player_name]
            advice["current_status"] = data["relationship"]
            
            # Analyze past interactions
            negative_count = sum(1 for i in data["interaction_history"] if i.get("reputation_change", 0) < 0)
            positive_count = sum(1 for i in data["interaction_history"] if i.get("reputation_change", 0) > 0)
            
            if data["relationship"] in ["hated", "disliked"]:
                advice["suggestions"].append("Consider apologizing for past negative interactions")
                advice["suggestions"].append("Offer help with a quest or dungeon to rebuild trust")
                advice["suggestions"].append("Make a fair trade offer to show good faith")
                
            elif data["relationship"] == "neutral":
                advice["suggestions"].append("Offer help with quests or dungeons")
                advice["suggestions"].append("Engage in friendly conversation")
                advice["suggestions"].append("Share useful information about game mechanics or quests")
                
            elif data["relationship"] in ["friendly", "trusted"]:
                advice["suggestions"].append("Invite to group activities regularly")
                advice["suggestions"].append("Share valuable resources or rare items")
                advice["suggestions"].append("Offer assistance with challenging content")
            
            # Add trading advice if they're a trade partner
            if player_name in self.trade_partners:
                trade_data = self.trade_partners[player_name]
                if trade_data["unfair_trades"] > 0:
                    advice["suggestions"].append("Make a particularly generous trade offer to compensate for past unfair trades")
                else:
                    advice["suggestions"].append("Continue making fair trades to maintain trust")
                    
        elif guild_name and guild_name in self.guilds:
            data = self.guilds[guild_name]
            advice["current_status"] = data["relationship"]
            
            if data["relationship"] in ["hated", "disliked"]:
                advice["suggestions"].append("Focus on building relationships with individual members first")
                advice["suggestions"].append("Assist guild members in group activities to rebuild trust")
                
            elif data["relationship"] == "neutral":
                advice["suggestions"].append("Join guild events and contribute positively")
                advice["suggestions"].append("Assist guild members with quests and dungeons")
                advice["suggestions"].append("Share valuable information with guild members")
                
            elif data["relationship"] in ["friendly", "trusted", "honored"]:
                advice["suggestions"].append("Consider donating useful items to the guild bank")
                advice["suggestions"].append("Take leadership roles in guild activities")
                advice["suggestions"].append("Help recruit new members for the guild")
                advice["suggestions"].append("Represent the guild positively in server-wide activities")
        
        # Add generic advice if no specific advice is available
        if not advice["suggestions"]:
            advice["suggestions"] = [
                "Engage in friendly conversation",
                "Offer assistance with quests or dungeons",
                "Make fair trades",
                "Share valuable information",
                "Invite to group activities"
            ]
        
        # Shuffle suggestions to provide variety
        random.shuffle(advice["suggestions"])
        
        return advice
    
    def detect_potential_scam(self, message: str, sender: str) -> Dict:
        """
        Analyze a message for potential scamming behaviors
        
        Args:
            message: The message to analyze
            sender: Who sent the message
            
        Returns:
            Dict: Scam detection results with confidence score and reasoning
        """
        result = {
            "is_scam": False,
            "confidence": 0,
            "reasoning": [],
            "sender_reputation": 0
        }
        
        # Get sender reputation if available
        if sender in self.players:
            result["sender_reputation"] = self.players[sender]["reputation"]
            
            # If sender already has very negative reputation, increase suspicion
            if self.players[sender]["reputation"] < self.relationship_thresholds["disliked"]:
                result["confidence"] += 10
                result["reasoning"].append("Sender has a history of negative interactions")
        
        # Look for common scam patterns
        scam_patterns = {
            r"free\s+gold": ("Promises of free gold", 40),
            r"double\s+your": ("Promises to double money/items", 50),
            r"password": ("Requests for account password", 90),
            r"account\s+details": ("Requests for account details", 80),
            r"log\s+out": ("Requests to log out and follow instructions", 60),
            r"website": ("Directs to external website", 30),
            r"click\s+this\s+link": ("Requests to click suspicious links", 70),
            r"your\s+account\s+will\s+be\s+banned": ("Threatens account ban", 60),
            r"gm": ("Impersonates a Game Master", 70),
            r"blizz": ("Impersonates Blizzard staff", 70),
            r"hack": ("Mentions hacks or cheats", 40),
            r"secret\s+method": ("Claims to have a 'secret method'", 30),
            r"urgent": ("Creates false urgency", 20),
            r"exclusive\s+offer": ("Claims exclusivity to pressure decision", 20),
            r"gold\s+seller": ("Mentions gold selling", 50),
            r"cheap\s+gold": ("Offers to sell gold cheaply", 60)
        }
        
        message_lower = message.lower()
        cumulative_confidence = 0
        
        for pattern, (reason, confidence) in scam_patterns.items():
            if re.search(pattern, message_lower):
                cumulative_confidence += confidence
                result["reasoning"].append(reason)
        
        # Check for trade-related scam patterns
        trade_scam_patterns = {
            r"cod": ("Cash on Delivery mention", 10),  # Not inherently scammy but worth noting
            r"trust\s+me": ("Explicit request for trust", 30),
            r"first\s+trade": ("Emphasizes it's a first trade", 15),
            r"i'll\s+send\s+after": ("Promises to send payment after", 40),
            r"send\s+first": ("Requests items before payment", 50)
        }
        
        for pattern, (reason, confidence) in trade_scam_patterns.items():
            if re.search(pattern, message_lower):
                cumulative_confidence += confidence
                result["reasoning"].append(reason)
        
        # Check message length - very short messages promising value are suspicious
        word_count = len(message_lower.split())
        if word_count < 5 and any(term in message_lower for term in ["gold", "free", "money", "cheap"]):
            cumulative_confidence += 15
            result["reasoning"].append("Suspiciously short message offering value")
        
        # Set final confidence score (cap at 100)
        result["confidence"] = min(100, cumulative_confidence)
        
        # Determine if it's a likely scam
        if result["confidence"] >= 70:
            result["is_scam"] = True
        
        return result
    
    def get_guild_contribution_strategy(self, guild_name: str) -> List[str]:
        """
        Generate strategies for contributing to a guild relationship
        
        Args:
            guild_name: Name of the guild
            
        Returns:
            List[str]: Contribution strategies
        """
        strategies = []
        
        # Get guild data if available
        guild_data = self.get_guild_relation(guild_name)
        guild_type = guild_data.get("guild_type", "unknown")
        
        # Generate strategies based on guild type
        if guild_type == "social":
            strategies = [
                "Participate actively in guild chat",
                "Join social events organized by the guild",
                "Help new guild members get oriented",
                "Share interesting discoveries or achievements",
                "Propose social activities for the guild"
            ]
        elif guild_type == "raiding":
            strategies = [
                "Maintain high attendance for scheduled raids",
                "Come prepared with appropriate consumables",
                "Research boss strategies ahead of time",
                "Help organize raid groups when needed",
                "Contribute to the guild's strategy development"
            ]
        elif guild_type == "pvp":
            strategies = [
                "Join guild PvP groups regularly",
                "Contribute to battleground strategies",
                "Help train newer PvP players",
                "Report important PvP developments on the server",
                "Represent the guild well in server PvP events"
            ]
        elif guild_type == "leveling":
            strategies = [
                "Help other members with difficult quests",
                "Share leveling tips and optimal routes",
                "Run lower level members through appropriate content",
                "Contribute to the guild knowledge base",
                "Craft leveling gear for guild members"
            ]
        else:
            # Generic strategies for unknown guild types
            strategies = [
                "Contribute regularly to the guild bank",
                "Participate in guild chat and activities",
                "Assist guild members with quests and dungeons",
                "Represent the guild positively in server activities",
                "Help with guild recruitment when possible"
            ]
        
        # Add relationship-specific strategies
        relationship = guild_data.get("relationship", "neutral")
        
        if relationship in ["hated", "disliked"]:
            strategies.append("Focus on rebuilding trust through consistent positive interactions")
            strategies.append("Apologize for past negative interactions if applicable")
            strategies.append("Start with small contributions to demonstrate reliability")
            
        elif relationship == "neutral":
            strategies.append("Increase visibility by participating more in guild activities")
            strategies.append("Find ways to be helpful to established guild members")
            
        elif relationship in ["friendly", "trusted"]:
            strategies.append("Take more leadership in guild activities")
            strategies.append("Mentor newer members to strengthen guild cohesion")
            strategies.append("Propose and organize guild events")
            
        elif relationship == "honored":
            strategies.append("Consider taking official leadership roles if available")
            strategies.append("Help shape guild policy and direction")
            strategies.append("Represent the guild in server-wide activities and leadership")
        
        # Randomize order for variety
        random.shuffle(strategies)
        
        return strategies[:5]  # Return top 5 strategies