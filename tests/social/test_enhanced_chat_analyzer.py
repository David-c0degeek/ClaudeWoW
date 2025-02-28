"""
Tests for the Enhanced Chat Analyzer Module
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import with error handling for CI environments
try:
    from src.social.enhanced_chat_analyzer import EnhancedChatAnalyzer, ChatContext, ChatMemory
    IMPORTS_SUCCEEDED = True
except ImportError:
    # Create mock classes for testing
    EnhancedChatAnalyzer = MagicMock
    ChatContext = MagicMock
    ChatMemory = MagicMock
    IMPORTS_SUCCEEDED = False


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Enhanced chat modules not available")
class TestEnhancedChatAnalyzer(unittest.TestCase):
    """Test cases for Enhanced Chat Analyzer module"""

    def setUp(self):
        """Set up the test environment"""
        # Mock config
        self.config = {
            "social": {
                "chat": {
                    "max_history": 50,
                    "embeddings_model": "text-embedding-3-small",
                    "response_model": "claude-3-haiku-20240307",
                    "harassment_threshold": 0.7,
                    "memory_retention_days": 7
                }
            }
        }
        
        # Mock knowledge
        self.knowledge = MagicMock()
        
        # Create analyzer with mocked LLM
        with patch('src.social.enhanced_chat_analyzer.LLMInterface'):
            self.analyzer = EnhancedChatAnalyzer(self.config, self.knowledge)
    
    def test_initialization(self):
        """Test that the analyzer initializes correctly"""
        self.assertIsInstance(self.analyzer, EnhancedChatAnalyzer)
        self.assertEqual(self.analyzer.max_history, 50)
        self.assertEqual(self.analyzer.harassment_threshold, 0.7)
        self.assertIsInstance(self.analyzer.chat_memory, ChatMemory)
        self.assertIsInstance(self.analyzer.chat_context, ChatContext)
    
    def test_analyze_message(self):
        """Test chat message analysis"""
        # Mock LLM response
        self.analyzer.llm.get_chat_completion.return_value = json.dumps({
            "intent": "request_information",
            "sentiment": "neutral",
            "harassment_score": 0.1,
            "requires_response": True,
            "context_relevance": "high",
            "social_impact": "neutral"
        })
        
        # Test message
        message = {
            "sender": "Player1",
            "content": "Hello, how do I get to Stormwind?",
            "channel": "whisper",
            "timestamp": "2023-06-15T14:30:00Z"
        }
        
        # Analyze message
        analysis = self.analyzer.analyze_message(message)
        
        # Check analysis results
        self.assertEqual(analysis["intent"], "request_information")
        self.assertEqual(analysis["sentiment"], "neutral")
        self.assertLess(analysis["harassment_score"], self.analyzer.harassment_threshold)
        self.assertTrue(analysis["requires_response"])
        
        # Verify chat context was updated
        self.assertIn(message, self.analyzer.chat_context.recent_messages)
    
    def test_detect_harassment(self):
        """Test harassment detection"""
        # Test obvious harassment
        message = {
            "sender": "BadPlayer",
            "content": "You're terrible at this game, uninstall now you loser!",
            "channel": "whisper",
            "timestamp": "2023-06-15T14:30:00Z"
        }
        
        # Mock high harassment score
        self.analyzer.llm.get_chat_completion.return_value = json.dumps({
            "intent": "harassment",
            "sentiment": "negative",
            "harassment_score": 0.9,
            "requires_response": False,
            "context_relevance": "none",
            "social_impact": "negative"
        })
        
        # Analyze harassment message
        analysis = self.analyzer.analyze_message(message)
        
        # Should be classified as harassment
        self.assertGreaterEqual(analysis["harassment_score"], self.analyzer.harassment_threshold)
        self.assertEqual(analysis["intent"], "harassment")
    
    def test_generate_response(self):
        """Test response generation"""
        # Mock analysis
        analysis = {
            "intent": "request_information",
            "sentiment": "neutral",
            "harassment_score": 0.1,
            "requires_response": True,
            "context_relevance": "high",
            "topic": "directions"
        }
        
        # Mock message
        message = {
            "sender": "Player1",
            "content": "How do I get to Ironforge from Stormwind?",
            "channel": "whisper",
            "timestamp": "2023-06-15T14:30:00Z"
        }
        
        # Mock LLM response for response generation
        mock_response = "Take the Deeprun Tram from Stormwind to Ironforge. It's located in the Dwarven District."
        self.analyzer.llm.get_chat_completion.return_value = mock_response
        
        # Generate response
        response = self.analyzer.generate_response(message, analysis)
        
        # Check response
        self.assertEqual(response, mock_response)
    
    def test_update_relationship(self):
        """Test relationship updating based on interactions"""
        # Initial interaction
        sender = "FriendlyPlayer"
        
        # Mock positive analysis
        analysis = {
            "intent": "social_greeting",
            "sentiment": "positive",
            "harassment_score": 0.0,
            "requires_response": True,
            "context_relevance": "high",
            "social_impact": "positive"
        }
        
        # Update relationship
        self.analyzer.update_relationship(sender, analysis)
        
        # Check relationship was created positively
        relationship = self.analyzer.relationships.get(sender, {})
        self.assertGreater(relationship.get("trust_score", 0), 0)
        self.assertEqual(relationship.get("interaction_count", 0), 1)
        
        # Test negative interaction
        analysis["sentiment"] = "negative"
        analysis["social_impact"] = "negative"
        analysis["harassment_score"] = 0.6
        
        # Update relationship again
        self.analyzer.update_relationship(sender, analysis)
        
        # Trust score should decrease
        updated_relationship = self.analyzer.relationships.get(sender, {})
        self.assertEqual(updated_relationship.get("interaction_count", 0), 2)
        self.assertLess(updated_relationship.get("trust_score", 0), relationship.get("trust_score", 0))
    
    def test_get_chat_history(self):
        """Test retrieving chat history with a specific player"""
        # Add messages to context
        player = "HistoryPlayer"
        self.analyzer.chat_context.recent_messages = [
            {
                "sender": player,
                "content": "Hello there!",
                "channel": "whisper",
                "timestamp": "2023-06-15T14:30:00Z"
            },
            {
                "sender": "OtherPlayer",
                "content": "Can you help me?",
                "channel": "whisper",
                "timestamp": "2023-06-15T14:31:00Z"
            },
            {
                "sender": player,
                "content": "How are you today?",
                "channel": "whisper",
                "timestamp": "2023-06-15T14:32:00Z"
            }
        ]
        
        # Get history with player
        history = self.analyzer.get_chat_history(player)
        
        # Should only contain messages from/to the specified player
        self.assertEqual(len(history), 2)
        self.assertTrue(all(msg["sender"] == player for msg in history))
    
    def test_save_and_load_memory(self):
        """Test saving and loading chat memory"""
        # Create a memory entry
        player = "MemoryPlayer"
        memory_item = {
            "sender": player,
            "content": "I'm looking for the best mining spots in Elwynn Forest.",
            "intent": "request_information",
            "timestamp": "2023-06-15T14:30:00Z",
            "topic": "mining",
            "sentiment": "neutral"
        }
        
        # Add to memory
        self.analyzer.chat_memory.add_memory(memory_item)
        
        # Mock io operations
        with patch('builtins.open', unittest.mock.mock_open()):
            with patch('json.dump') as mock_dump:
                # Save memory
                self.analyzer.save_memory("test_memory.json")
                
                # Check if memory was saved
                mock_dump.assert_called_once()
            
            with patch('json.load') as mock_load:
                # Mock loaded data
                mock_load.return_value = {"memories": [memory_item]}
                
                # Load memory
                self.analyzer.load_memory("test_memory.json")
                
                # Check if load was called
                mock_load.assert_called_once()


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Enhanced chat modules not available")
class TestChatContext(unittest.TestCase):
    """Test cases for the Chat Context class"""

    def setUp(self):
        """Set up the test environment"""
        self.max_history = 10
        self.context = ChatContext(max_history=self.max_history)
    
    def test_initialization(self):
        """Test that the context initializes correctly"""
        self.assertIsInstance(self.context, ChatContext)
        self.assertEqual(self.context.max_history, 10)
        self.assertEqual(len(self.context.recent_messages), 0)
        self.assertEqual(len(self.context.active_conversations), 0)
    
    def test_add_message(self):
        """Test adding messages to context"""
        message = {
            "sender": "TestPlayer",
            "content": "Hello there!",
            "channel": "whisper",
            "timestamp": "2023-06-15T14:30:00Z"
        }
        
        # Add message
        self.context.add_message(message)
        
        # Check message was added
        self.assertEqual(len(self.context.recent_messages), 1)
        self.assertEqual(self.context.recent_messages[0], message)
        
        # Check conversation was created
        self.assertIn("TestPlayer", self.context.active_conversations)
    
    def test_get_conversation(self):
        """Test retrieving conversation history with a player"""
        player = "ConversationPlayer"
        
        # Add several messages
        messages = [
            {
                "sender": player,
                "content": "Message 1",
                "channel": "whisper",
                "timestamp": "2023-06-15T14:30:00Z"
            },
            {
                "sender": player,
                "content": "Message 2",
                "channel": "whisper",
                "timestamp": "2023-06-15T14:31:00Z"
            },
            {
                "sender": "OtherPlayer",
                "content": "Unrelated",
                "channel": "whisper",
                "timestamp": "2023-06-15T14:32:00Z"
            }
        ]
        
        for msg in messages:
            self.context.add_message(msg)
        
        # Get conversation with player
        conversation = self.context.get_conversation(player)
        
        # Should only have messages from the specified player
        self.assertEqual(len(conversation), 2)
        self.assertTrue(all(msg["sender"] == player for msg in conversation))
    
    def test_get_channel_messages(self):
        """Test retrieving messages from a specific channel"""
        # Add messages from different channels
        messages = [
            {
                "sender": "Player1",
                "content": "Guild message",
                "channel": "guild",
                "timestamp": "2023-06-15T14:30:00Z"
            },
            {
                "sender": "Player2",
                "content": "Whisper message",
                "channel": "whisper",
                "timestamp": "2023-06-15T14:31:00Z"
            },
            {
                "sender": "Player3",
                "content": "Another guild message",
                "channel": "guild",
                "timestamp": "2023-06-15T14:32:00Z"
            }
        ]
        
        for msg in messages:
            self.context.add_message(msg)
        
        # Get guild messages
        guild_messages = self.context.get_channel_messages("guild")
        
        # Should only have guild messages
        self.assertEqual(len(guild_messages), 2)
        self.assertTrue(all(msg["channel"] == "guild" for msg in guild_messages))
    
    def test_history_limit(self):
        """Test that history respects the maximum size limit"""
        # Add more messages than the limit
        for i in range(self.max_history + 5):
            message = {
                "sender": f"Player{i}",
                "content": f"Message {i}",
                "channel": "whisper",
                "timestamp": f"2023-06-15T14:{i}:00Z"
            }
            self.context.add_message(message)
        
        # Check that only max_history messages are kept
        self.assertEqual(len(self.context.recent_messages), self.max_history)
        
        # Oldest messages should be removed (FIFO)
        oldest_index = 5  # Since we added max_history + 5 messages
        self.assertEqual(self.context.recent_messages[0]["content"], f"Message {oldest_index}")


@unittest.skipIf(not IMPORTS_SUCCEEDED, "Enhanced chat modules not available")
class TestChatMemory(unittest.TestCase):
    """Test cases for the Chat Memory class"""

    def setUp(self):
        """Set up the test environment"""
        self.retention_days = 7
        self.memory = ChatMemory(retention_days=self.retention_days)
    
    def test_initialization(self):
        """Test that the memory initializes correctly"""
        self.assertIsInstance(self.memory, ChatMemory)
        self.assertEqual(self.memory.retention_days, 7)
        self.assertEqual(len(self.memory.memories), 0)
    
    def test_add_memory(self):
        """Test adding items to memory"""
        memory_item = {
            "sender": "MemoryPlayer",
            "content": "I prefer gathering herbs over mining.",
            "intent": "share_preference",
            "timestamp": "2023-06-15T14:30:00Z",
            "topic": "professions",
            "sentiment": "positive"
        }
        
        # Add memory
        self.memory.add_memory(memory_item)
        
        # Check memory was added
        self.assertEqual(len(self.memory.memories), 1)
        self.assertEqual(self.memory.memories[0], memory_item)
    
    def test_get_memories_by_player(self):
        """Test retrieving memories related to a specific player"""
        player = "PlayerWithMemories"
        
        # Add memories from different players
        memories = [
            {
                "sender": player,
                "content": "Memory 1",
                "topic": "test",
                "timestamp": "2023-06-15T14:30:00Z"
            },
            {
                "sender": "OtherPlayer",
                "content": "Unrelated",
                "topic": "other",
                "timestamp": "2023-06-15T14:31:00Z"
            },
            {
                "sender": player,
                "content": "Memory 2",
                "topic": "test",
                "timestamp": "2023-06-15T14:32:00Z"
            }
        ]
        
        for mem in memories:
            self.memory.add_memory(mem)
        
        # Get memories for player
        player_memories = self.memory.get_memories_by_player(player)
        
        # Should only have memories from the specified player
        self.assertEqual(len(player_memories), 2)
        self.assertTrue(all(mem["sender"] == player for mem in player_memories))
    
    def test_get_memories_by_topic(self):
        """Test retrieving memories related to a specific topic"""
        # Add memories with different topics
        memories = [
            {
                "sender": "Player1",
                "content": "I like fishing in Booty Bay",
                "topic": "fishing",
                "timestamp": "2023-06-15T14:30:00Z"
            },
            {
                "sender": "Player2",
                "content": "The best ore is in Winterspring",
                "topic": "mining",
                "timestamp": "2023-06-15T14:31:00Z"
            },
            {
                "sender": "Player3",
                "content": "Fishing is relaxing in Elwynn Forest",
                "topic": "fishing",
                "timestamp": "2023-06-15T14:32:00Z"
            }
        ]
        
        for mem in memories:
            self.memory.add_memory(mem)
        
        # Get fishing-related memories
        fishing_memories = self.memory.get_memories_by_topic("fishing")
        
        # Should only have fishing memories
        self.assertEqual(len(fishing_memories), 2)
        self.assertTrue(all(mem["topic"] == "fishing" for mem in fishing_memories))
    
    def test_prune_old_memories(self):
        """Test pruning of old memories"""
        from datetime import datetime, timedelta
        
        # Create memories with different timestamps
        current_time = datetime.now().isoformat()
        old_time = (datetime.now() - timedelta(days=self.retention_days + 2)).isoformat()
        
        memories = [
            {
                "sender": "Player1",
                "content": "Recent memory",
                "topic": "test",
                "timestamp": current_time
            },
            {
                "sender": "Player2",
                "content": "Old memory",
                "topic": "test",
                "timestamp": old_time
            }
        ]
        
        for mem in memories:
            self.memory.add_memory(mem)
        
        # Prune old memories
        self.memory.prune_old_memories()
        
        # Should only have recent memories
        self.assertEqual(len(self.memory.memories), 1)
        self.assertEqual(self.memory.memories[0]["content"], "Recent memory")
    
    def test_search_memories(self):
        """Test searching memories by content"""
        # Add memories with different content
        memories = [
            {
                "sender": "Player1",
                "content": "I need help with the Deadmines dungeon",
                "topic": "dungeons",
                "timestamp": "2023-06-15T14:30:00Z"
            },
            {
                "sender": "Player2",
                "content": "Where is the auction house in Stormwind?",
                "topic": "locations",
                "timestamp": "2023-06-15T14:31:00Z"
            },
            {
                "sender": "Player3",
                "content": "The entrance to Deadmines is in Westfall",
                "topic": "dungeons",
                "timestamp": "2023-06-15T14:32:00Z"
            }
        ]
        
        for mem in memories:
            self.memory.add_memory(mem)
        
        # Search for "Deadmines"
        deadmines_memories = self.memory.search_memories("Deadmines")
        
        # Should find both memories mentioning Deadmines
        self.assertEqual(len(deadmines_memories), 2)
        for mem in deadmines_memories:
            self.assertIn("Deadmines", mem["content"])


if __name__ == "__main__":
    unittest.main()