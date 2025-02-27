# src/social/social_manager.py

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import threading
import random

from src.social.chat_analyzer import ChatAnalyzer
from src.social.group_coordinator import GroupCoordinator
from src.perception.screen_reader import GameState
from src.knowledge.game_knowledge import GameKnowledge
from src.social.llm_interface import LLMInterface
from src.social.character_profile import CharacterProfile

class SocialManager:
    """
    Manages all social interactions and coordination for the AI player
    """
    
    def __init__(self, config: Dict, knowledge_base: GameKnowledge):
        """
        Initialize the SocialManager
        
        Args:
            config: Configuration dictionary
            knowledge_base: Game knowledge base
        """
        self.logger = logging.getLogger("wow_ai.social.social_manager")
        self.config = config
        self.knowledge_base = knowledge_base
        
        # Initialize components
        self.chat_analyzer = ChatAnalyzer(config, knowledge_base)
        self.group_coordinator = GroupCoordinator(config, knowledge_base)
        
        # Social state
        self.chat_queue = []
        self.response_queue = []
        self.social_state = {
            "last_emote_time": 0,
            "last_greeting_time": 0,
            "last_group_chat_time": 0,
            "nearby_players": set(),
            "grouped_players": set(),
            "social_cooldowns": {}
        }
        
        # Chat throttling settings
        self.min_chat_interval = config.get("min_chat_interval", 5.0)  # Minimum seconds between messages
        self.chat_context_timeout = config.get("chat_context_timeout", 30.0)  # How long a conversation context lasts
        self.max_messages_per_minute = config.get("max_messages_per_minute", 8)  # Rate limit
        
        # Social behavior settings
        self.friendliness = config.get("social_friendliness", 0.5)  # 0.0 to 1.0
        self.chattiness = config.get("social_chattiness", 0.5)  # 0.0 to 1.0
        self.helpfulness = config.get("social_helpfulness", 0.8)  # 0.0 to 1.0
        
        # Initialize message history
        self.message_timestamps = []
        
        # LLM settings
        self.use_llm_for_group_chat = config.get("use_llm_for_group_chat", True)
        self.llm = LLMInterface(config)

        # Initialize character profile
        self.character_profile = CharacterProfile(config)

        self.logger.info("SocialManager initialized")
    
    def update(self, state: GameState) -> None:
        """
        Update social state based on game state
        
        Args:
            state: Current game state
        """
        # Update chat system
        if hasattr(state, "chat_log") and state.chat_log:
            self._process_chat_log(state.chat_log)
        
        # Update group coordination
        self.group_coordinator.update_group_state(state)
        
        # Update nearby players tracking
        if hasattr(state, "nearby_players") and state.nearby_players:
            current_players = set(player.get("name", "") for player in state.nearby_players)
            new_players = current_players - self.social_state["nearby_players"]
            
            # Consider greeting new players who come into view
            for player in new_players:
                if time.time() - self.social_state["last_greeting_time"] > 300:  # 5 minutes
                    if random.random() < self.friendliness * 0.3:  # Chance to greet based on friendliness
                        self.chat_queue.append({
                            "message": f"Hello, {player}!",
                            "channel": "say",
                            "priority": 0.5
                        })
                        self.social_state["last_greeting_time"] = time.time()
            
            # Update our tracking of nearby players
            self.social_state["nearby_players"] = current_players
        
        # Update grouped players tracking
        if hasattr(state, "group_members") and state.group_members:
            current_group = set(member.get("name", "") for member in state.group_members)
            new_members = current_group - self.social_state["grouped_players"]
            
            # Always greet new group members
            for player in new_members:
                self.chat_queue.append({
                    "message": f"Hi {player}, nice to group with you!",
                    "channel": "party",
                    "priority": 0.8
                })
            
            # Update our tracking of group members
            self.social_state["grouped_players"] = current_group
        
        # Consider emoting occasionally if friendliness is high
        if (self.friendliness > 0.7 and 
            time.time() - self.social_state["last_emote_time"] > 600 and  # 10 minutes
            random.random() < 0.2):
            self._consider_random_emote(state)
            self.social_state["last_emote_time"] = time.time()

        # Update character profile if needed
        if hasattr(state, "player_level") and state.player_level != self.character_profile.level:
            self.character_profile.level = state.player_level
            self._update_profile_achievements(state)
        
        # Consider greeting new players who come into view
        for player in new_players:
            if time.time() - self.social_state["last_greeting_time"] > 300:  # 5 minutes
                if random.random() < self.friendliness * 0.3:  # Chance to greet based on friendliness
                    # Use character profile for personalized greeting
                    relationship = "neutral"
                    if player in self.chat_analyzer.social_memory:
                        relationship = self.chat_analyzer.social_memory[player].get("relationship", "neutral")
                    
                    greeting = self.character_profile.get_greeting(player, relationship)
                    
                    self.chat_queue.append({
                        "message": greeting,
                        "channel": "say",
                        "priority": 0.5
                    })
                    self.social_state["last_greeting_time"] = time.time()
            
            # Update our tracking of nearby players
            self.social_state["nearby_players"] = current_players
        
        # Always greet new group members with personalized greeting
        for player in new_members:
            # Use character profile for personalized greeting
            greeting = self.character_profile.get_greeting(player, "friendly")
            
            self.chat_queue.append({
                "message": greeting,
                "channel": "party",
                "priority": 0.8
            })
        
        # Update our tracking of group members
        self.social_state["grouped_players"] = current_group
    
    def generate_social_actions(self, state: GameState) -> List[Dict]:
        """
        Generate social actions based on current state
        
        Args:
            state: Current game state
        
        Returns:
            List[Dict]: Social actions to take
        """
        actions = []
        
        # Check if we can send chat messages (rate limiting)
        current_time = time.time()
        self.message_timestamps = [t for t in self.message_timestamps if current_time - t < 60.0]
        can_send_chat = len(self.message_timestamps) < self.max_messages_per_minute
        
        # Process any pending chat responses
        if self.response_queue and can_send_chat:
            response = self.response_queue.pop(0)
            
            actions.append({
                "type": "chat",
                "message": response["message"],
                "channel": response["channel"],
                "description": "Send chat response"
            })
            
            self.message_timestamps.append(current_time)
        
        # Process any pending chat queue items
        elif self.chat_queue and can_send_chat:
            # Sort by priority
            self.chat_queue.sort(key=lambda x: x.get("priority", 0), reverse=True)
            
            chat_item = self.chat_queue.pop(0)
            
            actions.append({
                "type": "chat",
                "message": chat_item["message"],
                "channel": chat_item["channel"],
                "description": "Send chat message"
            })
            
            self.message_timestamps.append(current_time)
        
        # Get group coordination actions
        if state.is_in_group or state.is_in_raid:
            group_actions = self.group_coordinator.generate_coordination_actions(state)
            actions.extend(group_actions)
        
        # Consider occasional group chat if in a group
        if (state.is_in_group and 
            time.time() - self.social_state["last_group_chat_time"] > 300 and  # 5 minutes
            random.random() < self.chattiness * 0.2):  # Chance based on chattiness
            group_chat = self._generate_group_chat(state)
            if group_chat and can_send_chat:
                actions.append({
                    "type": "chat",
                    "message": group_chat,
                    "channel": "party",
                    "description": "Send casual group chat"
                })
                self.social_state["last_group_chat_time"] = time.time()
                self.message_timestamps.append(current_time)
        
        return actions
    
    def _process_chat_log(self, chat_log: List[str]) -> None:
        """
        Process new chat messages
        
        Args:
            chat_log: List of new chat messages
        """
        for chat_entry in chat_log:
            # Parse the chat entry
            sender, channel, message = self._parse_chat_entry(chat_entry)
            
            # Skip if we couldn't parse it
            if not sender or not message:
                continue
            
            # Skip if it's our own message
            player_name = self.config.get("player_name", "")
            if sender == player_name:
                continue
            
            # Analyze the message and generate a response
            response = self.chat_analyzer.analyze_chat(message, sender, channel)
            
            # If we have a response, add it to the queue
            if response:
                response_channel = channel
                
                # Use appropriate channel for the response
                if channel == "whisper":
                    response_channel = f"whisper:{sender}"
                elif channel == "say":
                    response_channel = "say"
                elif channel == "yell":
                    response_channel = "say"  # Respond to yells with normal say
                elif channel == "party" or channel == "raid":
                    response_channel = channel
                else:
                    # For other channels, whisper the person
                    response_channel = f"whisper:{sender}"
                
                self.response_queue.append({
                    "message": response,
                    "channel": response_channel,
                    "context": {
                        "original_message": message,
                        "sender": sender,
                        "timestamp": time.time()
                    }
                })
    
    def _parse_chat_entry(self, chat_entry: str) -> Tuple[str, str, str]:
        """
        Parse a chat log entry to extract sender, channel, and message
        
        Args:
            chat_entry: Raw chat log entry
        
        Returns:
            Tuple[str, str, str]: Sender, channel, and message
        """
        try:
            # This would need to match the format of the game's chat log
            # Format examples:
            # [Party] PlayerName: message
            # [Raid] PlayerName: message
            # PlayerName whispers: message
            # PlayerName says: message
            
            # Simple parsing for common formats
            if ']' in chat_entry:
                # Channel message
                channel_part, rest = chat_entry.split(']', 1)
                channel = channel_part.strip('[').lower()
                
                if ':' in rest:
                    sender_part, message = rest.split(':', 1)
                    sender = sender_part.strip()
                    return sender, channel, message.strip()
            elif 'whispers:' in chat_entry:
                # Whisper
                sender_part, message = chat_entry.split('whispers:', 1)
                sender = sender_part.strip()
                return sender, "whisper", message.strip()
            elif 'says:' in chat_entry:
                # Say
                sender_part, message = chat_entry.split('says:', 1)
                sender = sender_part.strip()
                return sender, "say", message.strip()
            elif 'yells:' in chat_entry:
                # Yell
                sender_part, message = chat_entry.split('yells:', 1)
                sender = sender_part.strip()
                return sender, "yell", message.strip()
            
            # If we can't parse it, return empty values
            return "", "", ""
            
        except Exception as e:
            self.logger.error(f"Error parsing chat entry: {e}")
            return "", "", ""
    
    def _consider_random_emote(self, state: GameState) -> None:
        """
        Consider using a random emote based on the situation
        
        Args:
            state: Current game state
        """
        # Simple emotes that work in most situations
        general_emotes = ["wave", "smile", "laugh", "cheer", "dance", "thank"]
        
        # Context-specific emotes
        if state.is_in_combat:
            context_emotes = ["charge", "battle", "roar", "threaten"]
        elif state.is_resting:
            context_emotes = ["sit", "sleep", "relax", "yawn"]
        elif state.is_in_group:
            context_emotes = ["greet", "salute", "thank", "hug"]
        else:
            context_emotes = ["wave", "hello", "curious", "flex"]
        
        # Combine and choose a random emote
        all_emotes = general_emotes + context_emotes
        chosen_emote = random.choice(all_emotes)
        
        # Add to chat queue with low priority
        self.chat_queue.append({
            "message": f"/{chosen_emote}",
            "channel": "emote",
            "priority": 0.2
        })
    
    def _get_recent_group_chat(self, state: GameState) -> str:
        """
        Get recent group chat history
        
        Args:
            state: Current game state
        
        Returns:
            str: Recent group chat
        """
        # Filter chat history for group messages
        group_chat = [entry for entry in self.chat_history 
                     if entry.get("channel") == "party" or entry.get("channel") == "raid"]
        
        # Get last 3 messages
        recent = group_chat[-3:] if len(group_chat) > 3 else group_chat
        
        # Format as text
        return "; ".join([f"{entry.get('sender')}: {entry.get('message')}" for entry in recent])
   
    def _update_profile_achievements(self, state: GameState) -> None:
        """
        Update character profile achievements based on current state
        
        Args:
            state: Current game state
        """
        # Add level-based achievements
        level = state.player_level
        achievements = self.character_profile.profile["background"]["achievements"]
        
        # Check for level milestones
        if level >= 10 and not any("basic abilities" in ach for ach in achievements):
            achievements.append(f"Mastered basic {self.character_profile.character_class} abilities")
        
        if level >= 20 and not any("regions" in ach for ach in achievements):
            faction = self.character_profile.profile["basics"]["faction"]
            achievements.append(f"Explored multiple regions of {faction} territory")
        
        if level >= 40 and not any("mount" in ach for ach in achievements):
            achievements.append("Learned to ride a mount")
        
        if level >= 60 and not any("pinnacle" in ach for ach in achievements):
            achievements.append("Reached the pinnacle of my abilities")
        
        # Check for dungeon completions
        if hasattr(state, "completed_dungeons") and state.completed_dungeons:
            if not any("dungeon" in ach for ach in achievements):
                achievements.append("Completed my first dungeon")
            
            if len(state.completed_dungeons) >= 5 and not any("veteran" in ach for ach in achievements):
                achievements.append("Became a dungeon veteran")
        
        # Check for PvP achievements
        if hasattr(state, "pvp_kills") and state.pvp_kills:
            if state.pvp_kills >= 100 and not any("warrior" in ach for ach in achievements):
                achievements.append("Proven myself as a formidable PvP warrior")
        
        # Update the profile
        self.character_profile.profile["background"]["achievements"] = achievements
        self.character_profile._save_profile()
    
    def _generate_group_chat(self, state: GameState) -> Optional[str]:
        """
        Generate casual group chat based on the current situation
        
        Args:
            state: Current game state
        
        Returns:
            Optional[str]: Chat message or None
        """
        if self.use_llm_for_group_chat:
            # Build context for LLM
            context = {
                "current_activity": "in a dungeon group" if state.is_in_instance else "in a group questing",
                "group_members": ", ".join([m.get("name", "") for m in state.group_members]),
                "relationship": "friendly",  # Default to friendly for group members
                "conversation_history": self._get_recent_group_chat(state),
                "character_profile": self.character_profile.get_profile_as_prompt(),
                "game_state": self._format_game_state_for_llm(state)
            }
            
            # Use LLM to generate natural group chat
            prompt = "Generate a friendly casual comment or question for your group in World of Warcraft"
            return self.llm.generate_chat_response(prompt, "group", "party", context)
        else:
            # Use character profile phrases instead of fixed templates
            return self.character_profile.get_random_phrase()
    
    def _format_game_state_for_llm(self, state: GameState) -> str:
        """
        Format relevant game state information for LLM context
        
        Args:
            state: Current game state
        
        Returns:
            str: Formatted game state description
        """
        parts = []
        
        # Add location info
        if hasattr(state, "current_zone") and state.current_zone:
            parts.append(f"Currently in {state.current_zone}")
            
            if hasattr(state, "current_subzone") and state.current_subzone:
                parts.append(f"in the {state.current_subzone} area")
        
        # Add instance info
        if hasattr(state, "is_in_instance") and state.is_in_instance:
            if hasattr(state, "current_instance") and state.current_instance:
                parts.append(f"Running the {state.current_instance} dungeon")
            else:
                parts.append("In a dungeon")
        
        # Add quest info
        if hasattr(state, "current_quest") and state.current_quest:
            quest_name = state.current_quest.get("title", "a quest")
            parts.append(f"Working on the '{quest_name}' quest")
        
        # Add combat info
        if hasattr(state, "is_in_combat") and state.is_in_combat:
            if hasattr(state, "target") and state.target:
                parts.append(f"Fighting {state.target}")
            else:
                parts.append("In combat")
        
        return ". ".join(parts)