# src/social/enhanced_chat_analyzer.py

import re
import logging
import time
import random
from typing import Dict, List, Tuple, Any, Optional

from src.social.llm_interface import LLMInterface
from src.social.character_profile import CharacterProfile
from src.social.reputation_manager import ReputationManager

class EnhancedChatAnalyzer:
    """
    Advanced chat analyzer with improved social intelligence capabilities
    
    Features:
    - Context-aware chat processing
    - Relationship-based personalization
    - Scam and harassment detection
    - Group coordination communication
    - Server community reputation tracking
    """
    
    def __init__(self, config: Dict, knowledge_base):
        """
        Initialize the EnhancedChatAnalyzer
        
        Args:
            config: Configuration dictionary
            knowledge_base: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.social.enhanced_chat_analyzer")
        self.config = config
        self.knowledge_base = knowledge_base
        
        # Initialize components
        self.llm = LLMInterface(config)
        self.character_profile = CharacterProfile(config)
        self.reputation_manager = ReputationManager(config)
        
        # Chat message categories (enhanced with social intent)
        self.message_types = {
            "greeting": [r"hi\b", r"hello\b", r"hey\b", r"greetings", r"sup\b", r"o/", r"hola", r"howdy"],
            "farewell": [r"bye\b", r"cya", r"good night", r"see ya", r"later", r"take care"],
            "group_invite": [r"inv", r"invite", r"group", r"party", r"wanna join", r"need.*for", r"lfm", r"looking for"],
            "quest_related": [r"quest", r"objective", r"kill", r"collect", r"where is", r"how do i", r"help.*with"],
            "help_request": [r"help", r"assist", r"need a hand", r"can someone", r"anyone.*help", r"stuck"],
            "trade_request": [r"trade", r"buy", r"sell", r"wts", r"wtb", r"gold", r"price", r"cost"],
            "guild_related": [r"guild", r"recruit", r"joining", r"gbank", r"guildies"],
            "dungeon_related": [r"dungeon", r"instance", r"raid", r"tank", r"heal", r"dps", r"looking for", r"group for"],
            "gratitude": [r"thanks", r"thank you", r"thx", r"tyvm", r"appreciate", r"ty"],
            "apology": [r"sorry", r"my bad", r"apologize", r"didn't mean", r"oops", r"mistake"],
            "complaint": [r"lag", r"buggy", r"broken", r"stupid", r"annoying", r"hate", r"wtf", r"fix"],
            "congratulations": [r"grats", r"congratulations", r"congrats", r"well done", r"nice job", r"gz"],
            "joke": [r"lol", r"haha", r"rofl", r"lmao", r"joke", r"funny", r"hehe"],
            "roleplaying": [r"/me", r"*", r"roleplay", r"rp"],
            "question": [r"\?$", r"^who", r"^what", r"^where", r"^when", r"^why", r"^how", r"anyone know"]
        }
        
        # Social intent classifiers (deeper understanding of message purpose)
        self.social_intents = {
            "friendship_building": [r"friend", r"add", r"battletag", r"play.*together", r"regularly", r"again sometime"],
            "status_signaling": [r"just got", r"new.*gear", r"level.*up", r"achievement", r"look at", r"check.*out"],
            "help_offering": [r"can help", r"i could", r"let me", r"i'll assist", r"need.*hand"],
            "information_sharing": [r"did you know", r"fyi", r"heads up", r"psa", r"announcement"],
            "opinion_seeking": [r"what do you think", r"is it good", r"worth it", r"better to", r"opinion"],
            "community_building": [r"everyone", r"all welcome", r"server", r"community", r"event"],
            "advice_giving": [r"should", r"recommend", r"suggest", r"try", r"better if", r"advice"],
            "emotional_support": [r"it's ok", r"don't worry", r"hang in there", r"you'll get", r"next time"]
        }
        
        # Harassment patterns for detection
        self.harassment_patterns = {
            "personal_attack": [r"noob", r"scrub", r"suck", r"trash", r"garbage", r"pathetic", r"useless"],
            "discriminatory": [r"racist", r"sexist", r"homophobic", r"slurs", r"bigot"],
            "threatening": [r"kill you", r"find you", r"hunt you", r"report", r"revenge"],
            "spamming": [r"(.)\\1{4,}", r"(.{3,})\\1{3,}", r"^.{1,2}$"],  # Repeated characters, repeated short strings
            "griefing_intent": [r"grief", r"ruin", r"gank", r"corpse camp", r"steal", r"ninja"],
            "obscene": [r"explicit sexual language"]  # Placeholder pattern, would include actual words in real implementation
        }
        
        # Response templates by category
        self.response_templates = self._load_response_templates()
        
        # Social memory cache
        self.player_context = {}
        self.social_memory = {}
        self.conversation_threads = {}
        
        # Chat history for context
        self.chat_history = []
        self.max_history = config.get("max_chat_history", 50)
        
        # Configure LLM usage
        self.use_llm_threshold = config.get("use_llm_threshold", 0.6)
        self.prioritize_llm_channels = ["whisper", "party", "guild", "officer"]
        
        # Behavioral settings
        self.politeness_level = config.get("politeness_level", 0.8)  # 0.0 to 1.0
        self.chattiness = config.get("chattiness", 0.5)  # 0.0 to 1.0
        self.humor_level = config.get("humor_level", 0.4)  # 0.0 to 1.0
        self.relationship_weight = config.get("relationship_weight", 0.7)  # How much relationships influence responses
        
        self.logger.info("EnhancedChatAnalyzer initialized with social intelligence capabilities")
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """
        Load chat response templates
        
        Returns:
            Dict[str, List[str]]: Templates by category
        """
        templates = {
            "greeting": [
                "Hey there!",
                "Hello!",
                "Hi! How's it going?",
                "Greetings!",
                "Hey, what's up?",
                "Hi {name}!",
                "Hello {name}, nice to see you!",
                "Hey {name}, how have you been?"
            ],
            "farewell": [
                "Take care!",
                "See you later!",
                "Goodbye for now.",
                "Catch you later!",
                "Until next time.",
                "Take it easy, {name}.",
                "Safe travels, {name}."
            ],
            "group_invite_accept": [
                "Sure, I'd love to join.",
                "Sounds good to me.",
                "I'd be happy to group up.",
                "Yeah, let's party up.",
                "I'd be glad to join forces."
            ],
            "group_invite_decline": [
                "Sorry, I'm busy with something right now.",
                "Thanks for the invite, but I'm working on something solo at the moment.",
                "I appreciate it, but I need to finish this quest first.",
                "Maybe later? I'm in the middle of something.",
                "I'll have to pass for now, but thanks for thinking of me."
            ],
            "quest_help": [
                "I think that quest objective is {location}.",
                "Have you tried {suggestion}?",
                "The {target} can be found near {location}.",
                "I did that one recently. You need to {requirement}.",
                "That quest requires you to {requirement} in {location}."
            ],
            "general_help": [
                "What do you need help with?",
                "I can try to help. What's the issue?",
                "What are you trying to do?",
                "I might be able to assist with that.",
                "How can I be of assistance?"
            ],
            "trade_response": [
                "I'm not looking to trade right now, but thanks.",
                "What are you looking to {trade_type}?",
                "How much are you asking for that?",
                "I might be interested, what's your price?",
                "Let me check if I have what you're looking for."
            ],
            "guild_response": [
                "Thanks, but I'm not looking for a guild at the moment.",
                "What kind of guild is it?",
                "What activities does your guild focus on?",
                "How many members do you have?",
                "What's your guild's focus? PvE, PvP, or social?"
            ],
            "dungeon_interest": [
                "What role are you looking for?",
                "What dungeon are you running?",
                "What's the group composition so far?",
                "I might be interested. What level is the dungeon?",
                "Do you need a {player_role} for that run?"
            ],
            "dungeon_offer": [
                "I can {role} for your group if you need.",
                "I'd be happy to join as {role}.",
                "I'm available to {role} if you still need one.",
                "I could help out as {role} for that run.",
                "Need a {role}? I'm available."
            ],
            "thank_response": [
                "No problem!",
                "You're welcome!",
                "Glad to help!",
                "Anytime!",
                "Happy to assist!",
                "My pleasure, {name}."
            ],
            "gratitude_response": [
                "You're welcome!",
                "Glad I could help.",
                "No problem at all.",
                "Happy to be of service.",
                "Anytime, {name}."
            ],
            "apology_acceptance": [
                "No worries.",
                "It's all good.",
                "Don't worry about it.",
                "No harm done.",
                "That's alright, happens to everyone."
            ],
            "congratulations_response": [
                "Thank you!",
                "Thanks a lot!",
                "Appreciate it!",
                "Thanks! I'm pretty happy about it.",
                "Thank you, {name}!"
            ],
            "joke_response": [
                "Haha!",
                "That's pretty funny.",
                "Lol!",
                "Good one!",
                ":)"
            ],
            "scam_warning": [
                "I should be careful about this. It seems suspicious.",
                "This looks like a potential scam. I'll avoid engaging.",
                "This request seems unsafe. Better to ignore it.",
                "Warning: This appears to be a scam attempt.",
                "I won't respond to this suspicious message."
            ],
            "harassment_response": [
                "I'd prefer to keep conversations respectful.",
                "I'll be moving on from this conversation.",
                "Let's try to maintain a positive atmosphere.",
                "(Ignoring inappropriate comment)",
                "I'll be adding this person to my ignore list."
            ],
            "unknown": [
                "Hmm, I'm not sure about that.",
                "Interesting. Tell me more.",
                "I see.",
                "Got it.",
                "Hmm, let me think about that."
            ]
        }
        
        # Add relationship-specific templates
        templates["friendly"] = [
            "Hey {name}! Great to see you again!",
            "Hello {name}! How have you been?",
            "{name}! Just the person I was hoping to run into!",
            "Hey there, {name}! Ready for another adventure?"
        ]
        
        templates["trusted"] = [
            "Hey {name}! Always a pleasure!",
            "{name}! Perfect timing - I could use a trusted friend.",
            "Well if it isn't {name}! How's my favorite adventurer?",
            "Hey {name}! I was just thinking about our last adventure together."
        ]
        
        templates["disliked"] = [
            "Hello.",
            "Yes?",
            "What do you need?",
            "Can I help you with something?"
        ]
        
        return templates
    
    def analyze_chat(self, message: str, sender: str, channel: str, game_state: Dict = None) -> Optional[str]:
        """
        Analyze a chat message and generate an appropriate response with social intelligence
        
        Args:
            message: Chat message text
            sender: Name of the sender
            channel: Chat channel (say, whisper, party, etc.)
            game_state: Current game state for context
            
        Returns:
            Optional[str]: Response message or None if no response needed
        """
        # Track message in history
        timestamp = time.time()
        
        message_entry = {
            "message": message,
            "sender": sender,
            "channel": channel,
            "timestamp": timestamp,
            "flagged_as_scam": False,
            "flagged_as_harassment": False
        }
        
        self.chat_history.append(message_entry)
        
        # Trim history if too long
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
        
        # Get player relationship data
        player_relation = self.reputation_manager.get_player_relation(sender)
        relationship = player_relation.get("relationship", "neutral")
        
        # Check for scams
        scam_result = self.reputation_manager.detect_potential_scam(message, sender)
        if scam_result["is_scam"]:
            self.logger.warning(f"Potential scam detected from {sender}: {message} (Confidence: {scam_result['confidence']}%)")
            message_entry["flagged_as_scam"] = True
            
            # Update sender reputation if highly confident
            if scam_result["confidence"] >= 80:
                self.reputation_manager.update_player_reputation(
                    sender, 
                    "scam_attempt",
                    {"note": f"Attempted scam: {message}", "magnitude": scam_result["confidence"] / 100}
                )
                
            # Don't respond to likely scams
            if scam_result["confidence"] > 70:
                return self._select_template("scam_warning")
        
        # Check for harassment
        harassment_result = self._detect_harassment(message)
        if harassment_result["is_harassment"]:
            self.logger.warning(f"Potential harassment detected from {sender}: {message} (Type: {harassment_result['type']})")
            message_entry["flagged_as_harassment"] = True
            
            # Update sender reputation
            self.reputation_manager.update_player_reputation(
                sender, 
                "verbal_abuse",
                {"note": f"Harassment ({harassment_result['type']}): {message}"}
            )
            
            # Don't engage with harassment
            if harassment_result["severity"] >= "medium":
                return self._select_template("harassment_response")
        
        # Don't respond to ourselves or system messages
        if sender == "self" or sender == "":
            return None
        
        # Get player context
        self._update_player_context(sender, message, channel, game_state)
        
        # Only always respond to whispers or when directly addressed in other channels
        player_name = self.config.get("player_name", "")
        direct_address = (
            f"@{player_name}" in message or 
            f"{player_name}," in message or 
            f"{player_name}:" in message or
            message.lower().startswith(player_name.lower())
        )
        
        # Calculate response probability based on channel, relationship, and chattiness
        response_probability = self._get_response_probability(channel, relationship, direct_address)
        
        # Check if we should respond
        if random.random() > response_probability:
            return None
        
        # Decide whether to use LLM or template-based response
        use_llm = self._should_use_llm(message, channel, relationship)
        
        # Get conversation history for context
        history = self._get_conversation_history(sender)
        
        # Generate response
        if use_llm:
            # Prepare context for LLM
            context = self._prepare_llm_context(sender, channel, relationship, game_state)
            
            # Generate response using LLM
            response = self.llm.generate_chat_response(message, sender, channel, context)
            
            # Verify the response is appropriate for the character
            if not self.character_profile.is_appropriate_response(message, response):
                context["additional_guidance"] = "Make sure your response matches your character's personality and values."
                response = self.llm.generate_chat_response(message, sender, channel, context)
            
            # Update reputation based on positive interaction
            self.reputation_manager.update_player_reputation(
                sender,
                "chat_reciprocation",
                {"note": "Engaged in meaningful conversation"}
            )
            
            return response
        else:
            # Use template-based approach
            message_category = self._categorize_message(message.lower())
            social_intent = self._identify_social_intent(message.lower())
            
            response = self._generate_template_response(message, sender, channel, message_category, social_intent, relationship)
            
            # Small reputation boost for simple chat interaction
            if channel in ["whisper", "party", "guild"] or direct_address:
                self.reputation_manager.update_player_reputation(
                    sender,
                    "chat_reciprocation",
                    {"magnitude": 0.5}  # Smaller magnitude for template response
                )
            
            return response
    
    def _get_response_probability(self, channel: str, relationship: str, direct_address: bool) -> float:
        """
        Calculate probability of responding to a message
        
        Args:
            channel: Chat channel
            relationship: Relationship with sender
            direct_address: Whether we were directly addressed
            
        Returns:
            float: Probability of responding (0.0 to 1.0)
        """
        # Base probabilities by channel
        channel_probabilities = {
            "whisper": 0.95,       # Almost always respond to whispers
            "say": 0.3,            # Respond to nearby chat sometimes
            "yell": 0.2,           # Respond to yells rarely
            "party": 0.7,          # Respond to party chat often
            "guild": 0.5,          # Respond to guild chat moderately
            "officer": 0.6,        # Respond to officer chat often
            "raid": 0.4,           # Respond to raid chat sometimes
            "battleground": 0.2,   # Respond to BG chat rarely
            "trade": 0.1,          # Respond to trade chat very rarely
            "general": 0.15        # Respond to general chat rarely
        }
        
        # Get base probability by channel (default to 0.1)
        base_prob = channel_probabilities.get(channel, 0.1)
        
        # Always respond to direct address in non-global channels
        if direct_address and channel not in ["trade", "general", "yell"]:
            base_prob = 0.95
        elif direct_address:  # Increase probability for direct address in global channels
            base_prob += 0.3
        
        # Adjust by relationship
        relationship_modifiers = {
            "hated": -0.5,
            "disliked": -0.3,
            "neutral": 0.0,
            "friendly": 0.2,
            "trusted": 0.3,
            "honored": 0.4
        }
        
        relationship_mod = relationship_modifiers.get(relationship, 0.0)
        
        # Apply chattiness setting
        chattiness_factor = self.chattiness * 0.5  # Scale chattiness influence
        
        # Calculate final probability
        final_prob = base_prob + relationship_mod + chattiness_factor
        
        # Clamp to valid range
        return max(0.0, min(1.0, final_prob))
    
    def _should_use_llm(self, message: str, channel: str, relationship: str) -> bool:
        """
        Determine whether to use LLM for response generation
        
        Args:
            message: Message text
            channel: Chat channel
            relationship: Relationship with sender
            
        Returns:
            bool: Whether to use LLM
        """
        # Always use LLM for specific channels
        if channel in self.prioritize_llm_channels:
            return True
        
        # Use LLM based on message complexity
        is_complex = len(message.split()) > 6 or "?" in message or "!" in message
        
        # Prioritize LLM for closer relationships
        relationship_priority = {
            "trusted": 0.9,
            "friendly": 0.7,
            "neutral": 0.5,
            "disliked": 0.3,
            "hated": 0.1
        }
        
        relationship_factor = relationship_priority.get(relationship, 0.5)
        
        # Calculate LLM probability
        llm_probability = self.use_llm_threshold
        
        if is_complex:
            llm_probability += 0.3
            
        llm_probability *= relationship_factor
        
        # LLM generation is more likely for complex messages and closer relationships
        return random.random() < llm_probability
    
    def _update_player_context(self, player: str, message: str, channel: str, game_state: Dict = None):
        """
        Update player context information
        
        Args:
            player: Player name
            message: Message text
            channel: Chat channel
            game_state: Current game state
        """
        # Initialize player context if needed
        if player not in self.player_context:
            self.player_context[player] = {
                "last_message": "",
                "last_channel": "",
                "last_timestamp": 0,
                "conversation_count": 0,
                "topics_discussed": set(),
                "current_activity": None,
                "group_status": None,
                "location": None
            }
        
        # Update context
        context = self.player_context[player]
        context["last_message"] = message
        context["last_channel"] = channel
        context["last_timestamp"] = time.time()
        context["conversation_count"] += 1
        
        # Extract topics from message
        topics = self._extract_topics(message)
        context["topics_discussed"].update(topics)
        
        # Update with game state if available
        if game_state:
            if "current_activity" in game_state:
                context["current_activity"] = game_state["current_activity"]
            
            if "location" in game_state:
                context["location"] = game_state["location"]
            
            if "group_status" in game_state:
                context["group_status"] = game_state["group_status"]
    
    def _extract_topics(self, message: str) -> List[str]:
        """
        Extract discussion topics from a message
        
        Args:
            message: Message text
            
        Returns:
            List[str]: Identified topics
        """
        topics = []
        
        # Simple topic extraction based on keywords
        topic_keywords = {
            "questing": ["quest", "objective", "complete", "turn in", "reward"],
            "combat": ["fight", "kill", "damage", "tank", "heal", "dps", "ability"],
            "gear": ["item", "weapon", "armor", "upgrade", "stat", "equip"],
            "dungeons": ["dungeon", "instance", "boss", "loot", "run"],
            "raids": ["raid", "guild", "organize", "schedule", "team"],
            "pvp": ["pvp", "battleground", "arena", "honor", "enemy", "alliance", "horde"],
            "economy": ["gold", "auction", "buy", "sell", "price", "trade", "bargain"],
            "crafting": ["craft", "profession", "recipe", "material", "make"],
            "social": ["guild", "friend", "group", "chat", "meet", "help"]
        }
        
        message_lower = message.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _categorize_message(self, message: str) -> str:
        """
        Categorize a message based on content patterns
        
        Args:
            message: Message to categorize
            
        Returns:
            str: Message category
        """
        for category, patterns in self.message_types.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return category
                    
        return "unknown"
    
    def _identify_social_intent(self, message: str) -> Optional[str]:
        """
        Identify the social intent behind a message
        
        Args:
            message: Message text
            
        Returns:
            Optional[str]: Identified social intent or None
        """
        for intent, patterns in self.social_intents.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return intent
        
        return None
    
    def _detect_harassment(self, message: str) -> Dict:
        """
        Detect potential harassment in a message
        
        Args:
            message: Message to analyze
            
        Returns:
            Dict: Harassment detection results
        """
        result = {
            "is_harassment": False,
            "type": None,
            "severity": "low",
            "evidence": []
        }
        
        message_lower = message.lower()
        
        # Check against harassment patterns
        for harass_type, patterns in self.harassment_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    result["is_harassment"] = True
                    result["type"] = harass_type
                    result["evidence"].append(pattern)
        
        # Determine severity based on evidence count and type
        if result["is_harassment"]:
            evidence_count = len(result["evidence"])
            
            if evidence_count >= 3 or result["type"] in ["discriminatory", "threatening"]:
                result["severity"] = "high"
            elif evidence_count >= 2 or result["type"] in ["personal_attack", "obscene"]:
                result["severity"] = "medium"
        
        return result
    
    def _get_conversation_history(self, player: str, max_entries: int = 10) -> List[Dict]:
        """
        Get recent conversation history with a specific player
        
        Args:
            player: Player name
            max_entries: Maximum number of history entries to return
            
        Returns:
            List[Dict]: Conversation history
        """
        history = []
        
        # Find recent messages to/from this player
        for entry in reversed(self.chat_history):
            if entry["sender"] == player or (entry["sender"] == "self" and "target" in entry and entry["target"] == player):
                history.append(entry)
                
                if len(history) >= max_entries:
                    break
        
        # Reverse back to chronological order
        return list(reversed(history))
    
    def _prepare_llm_context(self, sender: str, channel: str, relationship: str, game_state: Dict = None) -> Dict:
        """
        Prepare context information for LLM response generation
        
        Args:
            sender: Message sender
            channel: Chat channel
            relationship: Relationship with sender
            game_state: Current game state
            
        Returns:
            Dict: Context for LLM
        """
        # Get character profile as prompt
        character_profile_prompt = self.character_profile.get_profile_as_prompt()
        
        # Get conversation history
        history = self._get_conversation_history(sender)
        
        # Get player context
        player_ctx = self.player_context.get(sender, {})
        
        # Format relationship for LLM
        relationship_context = f"The player {sender} is {relationship} to me."
        if relationship in ["trusted", "friendly"]:
            relationship_context += " We have a positive history of interactions."
        elif relationship in ["disliked", "hated"]:
            relationship_context += " We have had some negative interactions in the past."
        
        # Get current activity
        current_activity = "exploring the world"
        if game_state and "current_activity" in game_state:
            current_activity = game_state["current_activity"]
        elif "current_activity" in player_ctx and player_ctx["current_activity"]:
            current_activity = player_ctx["current_activity"]
        
        # Build conversation style guidance based on settings
        style_guidance = []
        if self.politeness_level > 0.7:
            style_guidance.append("Be very polite and formal.")
        elif self.politeness_level < 0.3:
            style_guidance.append("Be casual and relaxed in tone.")
            
        if self.humor_level > 0.7:
            style_guidance.append("Use humor and wit when appropriate.")
        elif self.humor_level < 0.3:
            style_guidance.append("Keep responses straightforward and serious.")
        
        # Format style guidance
        style_guidance_text = " ".join(style_guidance)
        
        # Build final context
        context = {
            "conversation_history": history,
            "relationship": relationship_context,
            "current_activity": current_activity,
            "character_profile": character_profile_prompt,
            "style_guidance": style_guidance_text,
            "chat_channel": f"This message is in {channel} chat.",
            "topics_previously_discussed": list(player_ctx.get("topics_discussed", set()))
        }
        
        # Add game_state information if available
        if game_state:
            context["game_state"] = game_state
        
        return context
    
    def _generate_template_response(self, message: str, sender: str, channel: str, 
                                   category: str, social_intent: Optional[str], relationship: str) -> str:
        """
        Generate a template-based response
        
        Args:
            message: Original message
            sender: Message sender
            channel: Chat channel
            category: Message category
            social_intent: Identified social intent
            relationship: Relationship with sender
            
        Returns:
            str: Generated response
        """
        # Special handling for relationship-specific greetings
        if category == "greeting" and relationship in ["friendly", "trusted"]:
            return self._select_template(relationship).format(name=sender)
        
        # Handle based on message category
        if category == "greeting":
            return self._select_template("greeting").format(name=sender)
            
        elif category == "farewell":
            return self._select_template("farewell").format(name=sender)
            
        elif category == "group_invite":
            # Decision logic for accepting invites
            should_accept = relationship in ["friendly", "trusted"]
            template_key = "group_invite_accept" if should_accept else "group_invite_decline"
            return self._select_template(template_key)
            
        elif category == "quest_related":
            # Try to fill in quest knowledge if available
            template = self._select_template("quest_help")
            # In a real implementation, would get actual quest data from knowledge base
            placeholders = {
                "location": "[relevant location]",
                "suggestion": "[helpful suggestion]",
                "target": "[quest target]",
                "requirement": "[quest requirement]"
            }
            
            for key, value in placeholders.items():
                template = template.replace(f"{{{key}}}", value)
                
            return template
            
        elif category == "help_request":
            return self._select_template("general_help")
            
        elif category == "trade_request":
            template = self._select_template("trade_response")
            trade_type = "buy" if "wtb" in message.lower() else "sell"
            return template.replace("{trade_type}", trade_type)
            
        elif category == "guild_related":
            return self._select_template("guild_response")
            
        elif category == "dungeon_related":
            if "looking for" in message.lower() or "need" in message.lower():
                player_role = "tank"  # Would be determined from player state
                template = self._select_template("dungeon_offer")
                return template.replace("{role}", player_role)
            else:
                return self._select_template("dungeon_interest")
                
        elif category == "gratitude":
            return self._select_template("gratitude_response").format(name=sender)
            
        elif category == "apology":
            return self._select_template("apology_acceptance")
            
        elif category == "congratulations":
            return self._select_template("congratulations_response").format(name=sender)
            
        elif category == "joke":
            return self._select_template("joke_response")
            
        # Adjust for social intent if category is unknown but intent is clear
        elif social_intent and category == "unknown":
            if social_intent == "friendship_building":
                return f"I'd be happy to group up sometime, {sender}."
                
            elif social_intent == "help_offering":
                return "That's really kind of you to offer help."
                
            elif social_intent == "information_sharing":
                return "Thanks for sharing that information!"
                
            elif social_intent == "opinion_seeking":
                return "That's an interesting question. Let me think about it..."
        
        # Default response
        return self._select_template("unknown")
    
    def _select_template(self, template_key: str) -> str:
        """
        Select a response template from the given category
        
        Args:
            template_key: Template category key
            
        Returns:
            str: Selected template
        """
        templates = self.response_templates.get(template_key, self.response_templates["unknown"])
        return random.choice(templates)
    
    def update_relationship(self, player: str, event_type: str, context: Optional[Dict] = None) -> Dict:
        """
        Update relationship with a player
        
        Args:
            player: Player name
            event_type: Type of interaction
            context: Additional context
            
        Returns:
            Dict: Updated relationship information
        """
        # Forward to reputation manager
        rep_score = self.reputation_manager.update_player_reputation(player, event_type, context)
        
        # Get updated relationship data
        return self.reputation_manager.get_player_relation(player)
    
    def get_player_relationship_status(self, player: str) -> Dict:
        """
        Get relationship status with a player
        
        Args:
            player: Player name
            
        Returns:
            Dict: Relationship information
        """
        return self.reputation_manager.get_player_relation(player)
    
    def get_relationship_advice(self, player: str) -> Dict:
        """
        Get advice for improving relationship with a player
        
        Args:
            player: Player name
            
        Returns:
            Dict: Relationship advice
        """
        return self.reputation_manager.generate_relationship_advice(player_name=player)
    
    def get_guild_contribution_strategy(self, guild: str) -> List[str]:
        """
        Get strategies for contributing to a guild
        
        Args:
            guild: Guild name
            
        Returns:
            List[str]: Contribution strategies
        """
        return self.reputation_manager.get_guild_contribution_strategy(guild)
    
    def add_to_social_memory(self, player: str, key: str, value: Any):
        """
        Add/update information in social memory
        
        Args:
            player: Player name
            key: Memory key
            value: Memory value
        """
        if player not in self.social_memory:
            self.social_memory[player] = {}
            
        self.social_memory[player][key] = value
    
    def get_from_social_memory(self, player: str, key: str) -> Any:
        """
        Retrieve information from social memory
        
        Args:
            player: Player name
            key: Memory key
            
        Returns:
            Any: Retrieved value or None
        """
        if player in self.social_memory and key in self.social_memory[player]:
            return self.social_memory[player][key]
            
        return None