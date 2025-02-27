# src/social/llm_interface.py

import logging
import time
import json
import requests
from typing import Dict, List, Tuple, Any, Optional
import os
import threading
import queue
import importlib.util

class LLMInterface:
    """
    Interface for communicating with various LLM APIs for natural language generation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the LLM Interface
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.social.llm_interface")
        self.config = config
        
        # LLM configuration
        self.provider = config.get("llm_provider", "openai").lower()
        self.api_key = config.get(f"{self.provider}_api_key", 
                                 os.environ.get(f"{self.provider.upper()}_API_KEY", ""))
        self.model = config.get(f"{self.provider}_model", self._get_default_model())
        self.max_tokens = config.get("llm_max_tokens", 150)  # Keep responses concise for chat
        self.temperature = config.get("llm_temperature", 0.7)
        
        # Character profile for context
        self.character_class = config.get("player_class", "warrior")
        self.character_race = config.get("player_race", "human")
        self.character_level = config.get("player_level", 1)
        self.character_name = config.get("player_name", "Adventurer")
        self.character_personality = config.get("player_personality", "friendly and helpful")
        
        # Request queue for async processing
        self.request_queue = queue.Queue()
        self.response_queues = {}  # request_id -> response_queue
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        # Message cache to avoid duplicate requests
        self.message_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Check for local model dependencies
        if self.provider in ["ollama", "local"]:
            self._check_local_dependencies()

        # Token usage tracking
        self.tokens_used = 0
        self.token_callback = None
        self.daily_token_limit = config.get("daily_token_limit", 10000)
        self.reset_date = None
        
        self.logger.info(f"LLMInterface initialized with provider: {self.provider}, model: {self.model}")
    
    def _get_default_model(self) -> str:
        """
        Get the default model for the selected provider
        
        Returns:
            str: Default model name
        """
        defaults = {
            "openai": "gpt-3.5-turbo",
            "claude": "claude-3-haiku-20240307",
            "anthropic": "claude-3-haiku-20240307",
            "ollama": "llama2",
            "local": "mistral-7b-instruct",
            "google": "gemini-pro",
            "azure": "gpt-4",
            "mistral": "mistral-small"
        }
        return defaults.get(self.provider, "gpt-3.5-turbo")
    
    def _check_local_dependencies(self) -> None:
        """
        Check if dependencies for local models are available
        """
        required_packages = []
        
        if self.provider == "ollama":
            required_packages.append("ollama")
        elif self.provider == "local":
            required_packages.extend(["transformers", "torch"])
        
        missing_packages = []
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.warning(f"Missing dependencies for {self.provider}: {', '.join(missing_packages)}")
            self.logger.warning(f"Install required packages with: pip install {' '.join(missing_packages)}")
    
    def generate_chat_response(self, message: str, sender: str, 
                              channel: str, context: Dict = None) -> str:
        """
        Generate a natural language response to a chat message
        
        Args:
            message: The message to respond to
            sender: Name of the message sender
            channel: Chat channel
            context: Additional context information
        
        Returns:
            str: Generated response
        """
        # Build a cache key
        cache_key = f"{message}_{sender}_{channel}_{self.provider}"
        
        # Check cache first
        current_time = time.time()
        if cache_key in self.message_cache:
            cache_entry = self.message_cache[cache_key]
            if current_time - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["response"]
        
        # Create conversation context
        conversation_context = self._build_conversation_context(message, sender, channel, context)
        
        # Use synchronous request for simplicity
        response = self._send_request_sync(conversation_context)
        
        # Cache the response
        self.message_cache[cache_key] = {
            "response": response,
            "timestamp": current_time
        }
        
        return response
    
    def generate_chat_response_async(self, message: str, sender: str, 
                                    channel: str, context: Dict = None,
                                    callback: callable = None) -> str:
        """
        Generate a natural language response asynchronously
        
        Args:
            message: The message to respond to
            sender: Name of the message sender
            channel: Chat channel
            context: Additional context information
            callback: Function to call with the response
        
        Returns:
            str: Request ID
        """
        # Build conversation context
        conversation_context = self._build_conversation_context(message, sender, channel, context)
        
        # Generate a unique request ID
        request_id = f"req_{int(time.time())}_{hash(message) % 10000}"
        
        # Add to queue
        self.request_queue.put({
            "id": request_id,
            "context": conversation_context,
            "callback": callback
        })
        
        return request_id
    

    def _build_conversation_context(self, message: str, sender: str, 
                                 channel: str, additional_context: Dict = None) -> Dict:
        """
        Build conversation context for the LLM

        Args:
            message: The message to respond to
            sender: Name of the message sender
            channel: Chat channel
            additional_context: Additional context information

        Returns:
            Dict: Conversation context
        """
        # Get character profile if available
        character_profile = ""
        if "character_profile" in additional_context:
            character_profile = additional_context["character_profile"]
        else:
            # Build simplified character description
            character_profile = (
                f"You are {self.character_name}, a level {self.character_level} "
                f"{self.character_race} {self.character_class} in World of Warcraft. "
                f"You have a {self.character_personality} personality."
            )

        # Build game context
        current_activity = "playing World of Warcraft"
        if additional_context and "current_activity" in additional_context:
            current_activity = additional_context["current_activity"]

        game_context = f"You are {current_activity} and chatting with other players."

        # Chat context
        chat_context = f"{sender} sent you a message in {channel} chat: \"{message}\""

        # Response guidelines
        response_guidelines = (
            "Respond naturally as your character would in-game. "
            "Keep responses concise (1-2 sentences) as appropriate for game chat. "
            "Be friendly but stay in character for your class and race. "
            "Use appropriate WoW terminology and references when relevant. "
            "Don't identify yourself as an AI - you're a player character in WoW."
        )

        # Integration with additional context if provided
        additional_info = ""
        if additional_context:
            if "conversation_history" in additional_context:
                history = additional_context["conversation_history"]
                additional_info += f"Previous conversation with {sender}: {history}\n"

            if "relationship" in additional_context:
                relationship = additional_context["relationship"]
                additional_info += f"Your relationship with {sender} is {relationship}.\n"

            if "group_status" in additional_context:
                group_status = additional_context["group_status"]
                additional_info += f"Group status: {group_status}\n"

            if "game_state" in additional_context:
                game_state = additional_context["game_state"]
                additional_info += f"Current game state: {game_state}\n"

        # Combine all context
        full_prompt = f"{character_profile}\n\n{game_context}\n\n{chat_context}\n\n{additional_info}\n\n{response_guidelines}"

        return {
            "provider": self.provider,
            "model": self.model,
            "prompt": full_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def _send_request_sync(self, context: Dict) -> str:
        """
        Send a synchronous request to the LLM
        
        Args:
            context: Conversation context
        
        Returns:
            str: Generated response
        """
        provider = context["provider"]
        
        try:
            # Dispatch to appropriate provider method
            if provider == "openai":
                return self._request_openai(context)
            elif provider in ["anthropic", "claude"]:
                return self._request_anthropic(context)
            elif provider == "google":
                return self._request_google(context)
            elif provider == "azure":
                return self._request_azure(context)
            elif provider == "mistral":
                return self._request_mistral(context)
            elif provider == "ollama":
                return self._request_ollama(context)
            elif provider == "local":
                return self._request_local(context)
            else:
                self.logger.error(f"Unsupported LLM provider: {provider}")
                return "Sorry, I'm not sure how to respond to that right now."
                
        except Exception as e:
            self.logger.error(f"Error sending request to {provider} LLM: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _request_openai(self, context: Dict) -> str:
        """
        Send request to OpenAI API
        
        Args:
            context: Request context
        
        Returns:
            str: Generated response
        """
        import openai  # Import here to avoid dependency if not used
        
        openai.api_key = self.api_key
        
        try:
            response = openai.ChatCompletion.create(
                model=context["model"],
                messages=[
                    {"role": "system", "content": "You are a World of Warcraft player character."},
                    {"role": "user", "content": context["prompt"]}
                ],
                max_tokens=context["max_tokens"],
                temperature=context["temperature"]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _request_anthropic(self, context: Dict) -> str:
        """
        Send request to Anthropic (Claude) API
        
        Args:
            context: Request context
        
        Returns:
            str: Generated response
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": context["model"],
                "messages": [
                    {"role": "user", "content": context["prompt"]}
                ],
                "max_tokens": context["max_tokens"],
                "temperature": context["temperature"]
            }
            
            response = requests.post("https://api.anthropic.com/v1/messages", 
                                    headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json["content"][0]["text"].strip()
            else:
                self.logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return "Sorry, I'm not sure how to respond to that right now."
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _request_google(self, context: Dict) -> str:
        """
        Send request to Google (Gemini) API
        
        Args:
            context: Request context
        
        Returns:
            str: Generated response
        """
        try:
            import google.generativeai as genai  # Import here to avoid dependency if not used
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(context["model"])
            
            response = model.generate_content(context["prompt"])
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Google API error: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _request_azure(self, context: Dict) -> str:
        """
        Send request to Azure OpenAI API
        
        Args:
            context: Request context
        
        Returns:
            str: Generated response
        """
        try:
            from openai import AzureOpenAI  # Import here to avoid dependency if not used
            
            client = AzureOpenAI(
                azure_endpoint=self.config.get("azure_endpoint", ""),
                api_key=self.api_key,
                api_version=self.config.get("azure_api_version", "2023-05-15")
            )
            
            response = client.chat.completions.create(
                model=context["model"],
                messages=[
                    {"role": "system", "content": "You are a World of Warcraft player character."},
                    {"role": "user", "content": context["prompt"]}
                ],
                max_tokens=context["max_tokens"],
                temperature=context["temperature"]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Azure API error: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _request_mistral(self, context: Dict) -> str:
        """
        Send request to Mistral AI API
        
        Args:
            context: Request context
        
        Returns:
            str: Generated response
        """
        try:
            import mistralai.client  # Import here to avoid dependency if not used
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            
            client = MistralClient(api_key=self.api_key)
            
            messages = [
                ChatMessage(role="user", content=context["prompt"])
            ]
            
            chat_response = client.chat(
                model=context["model"],
                messages=messages,
                max_tokens=context["max_tokens"],
                temperature=context["temperature"]
            )
            
            return chat_response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Mistral API error: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _request_ollama(self, context: Dict) -> str:
        """
        Send request to Ollama (local API)
        
        Args:
            context: Request context
        
        Returns:
            str: Generated response
        """
        try:
            import ollama  # Import here to avoid dependency if not used
            
            response = ollama.chat(
                model=context["model"],
                messages=[
                    {"role": "user", "content": context["prompt"]}
                ]
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _request_local(self, context: Dict) -> str:
        """
        Run inference on a local model
        
        Args:
            context: Request context
        
        Returns:
            str: Generated response
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Import here to avoid dependency
            import torch
            
            # Load model and tokenizer (with caching)
            if not hasattr(self, 'local_model') or not hasattr(self, 'local_tokenizer'):
                self.logger.info(f"Loading local model: {context['model']}")
                
                self.local_tokenizer = AutoTokenizer.from_pretrained(context["model"])
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    context["model"],
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                self.local_pipeline = pipeline(
                    "text-generation",
                    model=self.local_model,
                    tokenizer=self.local_tokenizer
                )
            
            # Generate response
            prompt = f"### Instruction:\n{context['prompt']}\n\n### Response:"
            
            result = self.local_pipeline(
                prompt,
                max_new_tokens=context["max_tokens"],
                temperature=context["temperature"],
                do_sample=True,
                return_full_text=False
            )
            
            return result[0]["generated_text"].strip()
            
        except Exception as e:
            self.logger.error(f"Local model error: {e}")
            return "Sorry, I'm not sure how to respond to that right now."
    
    def _process_queue(self) -> None:
        """
        Process the request queue in a background thread
        """
        while True:
            try:
                # Get a request from the queue
                request = self.request_queue.get()
                
                # Process the request
                response = self._send_request_sync(request["context"])
                
                # If callback provided, call it
                if request.get("callback"):
                    try:
                        request["callback"](response)
                    except Exception as e:
                        self.logger.error(f"Error in callback: {e}")
                
                # Mark as done
                self.request_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
            
            # Brief pause to prevent CPU hogging
            time.sleep(0.1)

    def set_token_callback(self, callback: callable) -> None:
        """
        Set callback function for token usage updates
        
        Args:
            callback: Function to call with token count
        """
        self.token_callback = callback
        
        # Initial callback
        if self.token_callback:
            self.token_callback(self.tokens_used)
    
    def _update_token_count(self, tokens: int) -> None:
        """
        Update token usage count
        
        Args:
            tokens: Number of tokens to add
        """
        self.tokens_used += tokens
        
        # Check for reset
        current_date = time.strftime("%Y-%m-%d")
        if not self.reset_date or self.reset_date != current_date:
            self.reset_date = current_date
            self.tokens_used = tokens  # Reset to current usage
        
        # Call callback if set
        if self.token_callback:
            self.token_callback(self.tokens_used)
        
        # Check limit
        if self.tokens_used > self.daily_token_limit:
            self.logger.warning(f"Daily token limit reached: {self.tokens_used}/{self.daily_token_limit}")
            # Could implement throttling or fallback to templates here