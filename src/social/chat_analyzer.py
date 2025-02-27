# src/social/chat_analyzer.py

import re
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from src.social.llm_interface import LLMInterface
from src.social.character_profile import CharacterProfile

import random

class ChatAnalyzer:
    """
    Analyzes and generates responses for in-game chat
    """
    
    def __init__(self, config: Dict, knowledge_base):
        """
        Initialize the ChatAnalyzer
        
        Args:
            config: Configuration dictionary
            knowledge_base: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.social.chat_analyzer")
        self.config = config
        self.knowledge_base = knowledge_base
        
        # Chat message categories
        self.message_types = {
            "greeting": [r"hi\b", r"hello\b", r"hey\b", r"greetings", r"sup\b", r"o/"],
            "group_invite": [r"inv", r"invite", r"group", r"party", r"wanna join"],
            "quest_related": [r"quest", r"objective", r"kill", r"collect", r"where is", r"how do i"],
            "help_request": [r"help", r"assist", r"need a hand", r"can someone"],
            "trade_request": [r"trade", r"buy", r"sell", r"wts", r"wtb", r"gold"],
            "guild_related": [r"guild", r"recruit", r"joining", r"gbank"],
            "dungeon_related": [r"dungeon", r"instance", r"raid", r"tank", r"heal", r"dps", r"looking for"]
        }
        
        # Response templates by category
        self.response_templates = self._load_response_templates()
        
        # Chat history for context
        self.chat_history = []
        self.max_history = 20
        
        # Social memory - remember players we've interacted with
        self.social_memory = {}
        
        self.logger.info("ChatAnalyzer initialized")
        
        # Initialize LLM interface
        self.llm = LLMInterface(config)
        
        # Configure when to use LLM vs templates
        self.use_llm_threshold = config.get("use_llm_threshold", 0.5)  # Probability threshold
        self.prioritize_llm_channels = ["whisper", "party", "guild"]  # Always use LLM for these
        
        # Initialize character profile
        self.character_profile = CharacterProfile(config)

        self.logger.info("ChatAnalyzer initialized with LLM support")
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """
        Load chat response templates
        
        Returns:
            Dict[str, List[str]]: Templates by category
        """
        # These could be loaded from a file for easier editing
        return {
            "greeting": [
                "Hey there!",
                "Hello!",
                "Hi! How's it going?",
                "Greetings!",
                "Hey, what's up?"
            ],
            "group_invite_accept": [
                "Sure, I'd love to join.",
                "Sounds good to me.",
                "I'd be happy to group up.",
                "Yeah, let's party up."
            ],
            "group_invite_decline": [
                "Sorry, I'm busy with something right now.",
                "Thanks for the invite, but I'm working on something solo at the moment.",
                "I appreciate it, but I need to finish this quest first.",
                "Maybe later? I'm in the middle of something."
            ],
            "quest_help": [
                "I think that quest objective is {location}.",
                "Have you tried {suggestion}?",
                "The {target} can be found near {location}.",
                "I did that one recently. You need to {requirement}."
            ],
            "general_help": [
                "What do you need help with?",
                "I can try to help. What's the issue?",
                "What are you trying to do?",
                "I might be able to assist with that."
            ],
            "trade_response": [
                "I'm not looking to trade right now, but thanks.",
                "What are you looking to {trade_type}?",
                "How much are you asking for that?",
                "I might be interested, what's your price?"
            ],
            "guild_response": [
                "Thanks, but I'm not looking for a guild at the moment.",
                "What kind of guild is it?",
                "What activities does your guild focus on?",
                "How many members do you have?"
            ],
            "dungeon_interest": [
                "What role are you looking for?",
                "What dungeon are you running?",
                "What's the group composition so far?",
                "I might be interested. What level is the dungeon?"
            ],
            "dungeon_offer": [
                "I can {role} for your group if you need.",
                "I'd be happy to join as {role}.",
                "I'm available to {role} if you still need one.",
                "I could help out as {role} for that run."
            ],
            "thank_response": [
                "No problem!",
                "You're welcome!",
                "Glad to help!",
                "Anytime!",
                "Happy to assist!"
            ],
            "unknown": [
                "Hmm, I'm not sure about that.",
                "Interesting. Tell me more.",
                "I see.",
                "Got it."
            ]
        }
    
    def analyze_chat(self, message: str, sender: str, channel: str) -> Optional[str]:
        """
        Analyze a chat message and generate an appropriate response
        
        Args:
            message: Chat message text
            sender: Name of the sender
            channel: Chat channel (say, whisper, party, etc.)
        
        Returns:
            Optional[str]: Response message or None if no response needed
        """
        # Add to chat history
        timestamp = time.time()
        self.chat_history.append({
            "message": message,
            "sender": sender,
            "channel": channel,
            "timestamp": timestamp
        })
        
        # Trim history if too long
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
        
        # Update social memory
        if sender not in self.social_memory:
            self.social_memory[sender] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "interactions": 0,
                "relationship": "neutral",  # neutral, friendly, unfriendly
                "notes": []
            }
        
        self.social_memory[sender]["last_seen"] = timestamp
        self.social_memory[sender]["interactions"] += 1
        
        # Don't respond to ourselves or system messages
        if sender == "self" or sender == "":
            return None
        
        # Only always respond to whispers or when directly addressed in other channels
        player_name = self.config.get("player_name", "")
        direct_address = f"@{player_name}" in message or f"{player_name}," in message or f"{player_name}:" in message
        
        if channel != "whisper" and not direct_address and random.random() > 0.2:
            # Only respond to 20% of general chat that doesn't address us directly
            return None
        
        # Decide whether to use LLM or template-based response
        use_llm = False
        
        # Always use LLM for specific channels
        if channel in self.prioritize_llm_channels:
            use_llm = True
        # Use LLM based on complexity of message
        elif len(message.split()) > 5 or "?" in message:
            use_llm = True
        # Use LLM based on probability threshold
        elif random.random() < self.use_llm_threshold:
            use_llm = True
        
        # Get conversation history for context
        history = self._get_conversation_history(sender)
        
        # Get relationship status
        relationship = self.social_memory[sender].get("relationship", "neutral")
        
        # Get character profile as prompt
        character_profile_prompt = self.character_profile.get_profile_as_prompt()
        
        # Get current activity description
        current_activity = self._get_current_activity()
        
        # Build additional context
        context = {
            "conversation_history": history,
            "relationship": relationship,
            "current_activity": current_activity,
            "character_profile": character_profile_prompt
        }
        
        if use_llm:
            response = self.llm.generate_chat_response(message, sender, channel, context)
            
            # Verify the response is appropriate for the character
            if not self.character_profile.is_appropriate_response(message, response):
                # If not appropriate, try again with more guidance
                context["additional_guidance"] = "Make sure your response matches your character's personality and values."
                response = self.llm.generate_chat_response(message, sender, channel, context)
            
            return response
        else:
            # Use existing template-based approach
            message_type = self._categorize_message(message.lower())
            return self._generate_response(message, sender, channel, message_type)
    
    def _categorize_message(self, message: str) -> str:
        """
        Categorize a message based on content
        
        Args:
            message: Message to categorize
        
        Returns:
            str: Message category
        """
        for category, patterns in self.message_types.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    return category
        
        # Check for thanks
        if any(word in message for word in ["thanks", "thank you", "ty", "thx"]):
            return "thank"
        
        return "unknown"
    
    def _generate_response(self, message: str, sender: str, channel: str, message_type: str) -> str:
        """
        Generate a response based on message type
        
        Args:
            message: Original message
            sender: Message sender
            channel: Chat channel
            message_type: Categorized message type
        
        Returns:
            str: Generated response
        """
        # Simple response based on message type
        if message_type == "greeting":
            # Pick a random greeting
            return random.choice(self.response_templates["greeting"])
        
        elif message_type == "group_invite":
            # Decide whether to accept based on current state and social memory
            player_relationship = self.social_memory[sender].get("relationship", "neutral")
            
            if player_relationship == "friendly" or random.random() > 0.5:
                return random.choice(self.response_templates["group_invite_accept"])
            else:
                return random.choice(self.response_templates["group_invite_decline"])
        
        elif message_type == "quest_related":
            # Try to extract quest information
            quest_info = self._extract_quest_info(message)
            
            if quest_info:
                template = random.choice(self.response_templates["quest_help"])
                return template.format(**quest_info)
            else:
                return random.choice(self.response_templates["general_help"])
        
        elif message_type == "help_request":
            return random.choice(self.response_templates["general_help"])
        
        elif message_type == "trade_request":
            trade_type = "buy" if "wtb" in message.lower() or "buy" in message.lower() else "sell"
            template = random.choice(self.response_templates["trade_response"])
            return template.format(trade_type=trade_type)
        
        elif message_type == "guild_related":
            return random.choice(self.response_templates["guild_response"])
        
        elif message_type == "dungeon_related":
            player_class = self.config.get("player_class", "")
            role = self._get_class_role(player_class)
            
            if "lfm" in message.lower() or "looking for" in message.lower():
                template = random.choice(self.response_templates["dungeon_interest"])
                return template
            else:
                template = random.choice(self.response_templates["dungeon_offer"])
                return template.format(role=role)
        
        elif message_type == "thank":
            return random.choice(self.response_templates["thank_response"])
        
        else:  # unknown
            return random.choice(self.response_templates["unknown"])
    
    def _extract_quest_info(self, message: str) -> Optional[Dict[str, str]]:
        """
        Extract quest information from a message
        
        Args:
            message: Message to analyze
        
        Returns:
            Optional[Dict[str, str]]: Extracted quest information or None
        """
        # This would be much more sophisticated in a real implementation
        # using NLP techniques to extract the quest name, objectives, etc.
        
        # Try to find quest name in message
        quest_match = re.search(r"[\"']([^\"']+)[\"']", message)
        if quest_match:
            quest_name = quest_match.group(1)
            
            # Check if we have knowledge about this quest
            quest_info = self.knowledge_base.get_quest_by_name(quest_name)
            
            if quest_info:
                return {
                    "location": quest_info.get("location", "unknown location"),
                    "target": quest_info.get("target", "unknown target"),
                    "requirement": quest_info.get("requirement", "complete the objectives"),
                    "suggestion": quest_info.get("suggestion", "checking the quest log for details")
                }
        
        # Generic response if we can't extract specific info
        return {
            "location": "marked on your map",
            "target": "quest objective",
            "requirement": "follow the quest markers",
            "suggestion": "checking your quest log for more details"
        }
    
    def _get_class_role(self, player_class: str) -> str:
        """
        Get the typical role for a class
        
        Args:
            player_class: Player class name
        
        Returns:
            str: Role (tank, healer, dps)
        """
        # This is simplistic - many classes can perform multiple roles
        tanks = ["warrior", "paladin", "druid", "death knight"]
        healers = ["priest", "paladin", "druid", "shaman"]
        
        player_class = player_class.lower()
        
        if player_class in tanks:
            return "tank"
        elif player_class in healers:
            return "heal"
        else:
            return "dps"
    
    def update_relationship(self, player_name: str, interaction_type: str, value: int = 1) -> None:
        """
        Update relationship status with a player
        
        Args:
            player_name: Name of the player
            interaction_type: Type of interaction (positive, negative, neutral)
            value: Value to adjust relationship by
        """
        if player_name not in self.social_memory:
            return
        
        # Get current relationship points (internal tracking)
        relationship_points = self.social_memory[player_name].get("relationship_points", 0)
        
        # Adjust points based on interaction
        if interaction_type == "positive":
            relationship_points += value
        elif interaction_type == "negative":
            relationship_points -= value
        
        # Update relationship status
        if relationship_points > 3:
            self.social_memory[player_name]["relationship"] = "friendly"
        elif relationship_points < -3:
            self.social_memory[player_name]["relationship"] = "unfriendly"
        else:
            self.social_memory[player_name]["relationship"] = "neutral"
        
        # Store updated points
        self.social_memory[player_name]["relationship_points"] = relationship_points
    
    def add_player_note(self, player_name: str, note: str) -> None:
        """
        Add a note about a player
        
        Args:
            player_name: Name of the player
            note: Note to add
        """
        if player_name not in self.social_memory:
            self.social_memory[player_name] = {
                "first_seen": time.time(),
                "last_seen": time.time(),
                "interactions": 0,
                "relationship": "neutral",
                "notes": []
            }
        
        self.social_memory[player_name]["notes"].append({
            "timestamp": time.time(),
            "note": note
        })
    
    def _get_conversation_history(self, sender: str) -> str:
        """
        Get recent conversation history with a sender
        
        Args:
            sender: Name of the sender
        
        Returns:
            str: Recent conversation history
        """
        # Look for recent messages from/to this sender
        relevant_history = []
        
        for entry in reversed(self.chat_history):
            if entry["sender"] == sender or entry["sender"] == "self" and entry["target"] == sender:
                relevant_history.insert(0, f"{entry['sender']}: {entry['message']}")
            
            # Limit to last 3 messages
            if len(relevant_history) >= 3:
                break
        
        return "; ".join(relevant_history)
    
    def _get_current_activity(self) -> str:
        """
        Get a description of current activity
        
        Returns:
            str: Current activity description
        """
        # This would normally use game state
        # For now, return a placeholder
        return "adventuring in Azeroth"