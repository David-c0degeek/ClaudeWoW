# src/utils/llm_config.py

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from typing import Dict, Callable

class LLMConfigPanel:
    """
    Configuration panel for LLM settings
    """
    
    def __init__(self, parent, config: Dict, save_callback: Callable):
        """
        Initialize the LLM configuration panel
        
        Args:
            parent: Parent widget
            config: Current configuration
            save_callback: Callback function to save configuration
        """
        self.parent = parent
        self.config = config
        self.save_callback = save_callback
        
        self.frame = ttk.LabelFrame(parent, text="LLM Configuration")
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Provider selection
        ttk.Label(self.frame, text="LLM Provider:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.provider_var = tk.StringVar(value=config.get("llm_provider", "openai"))
        provider_options = ["openai", "claude", "google", "azure", "mistral", "ollama", "local"]
        self.provider_dropdown = ttk.Combobox(self.frame, textvariable=self.provider_var, values=provider_options)
        self.provider_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.provider_dropdown.bind("<<ComboboxSelected>>", self._update_model_options)
        
        # Model selection
        ttk.Label(self.frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.model_var = tk.StringVar(value=self._get_current_model())
        self.model_dropdown = ttk.Combobox(self.frame, textvariable=self.model_var)
        self.model_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self._update_model_options()
        
        # API Key
        ttk.Label(self.frame, text="API Key:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.api_key_var = tk.StringVar(value=self._get_current_api_key())
        self.api_key_entry = ttk.Entry(self.frame, textvariable=self.api_key_var, width=40, show="*")
        self.api_key_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Show/hide API key button
        self.show_key = tk.BooleanVar(value=False)
        self.show_key_btn = ttk.Checkbutton(self.frame, text="Show Key", variable=self.show_key, 
                                           command=self._toggle_key_visibility)
        self.show_key_btn.grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Advanced settings section
        advanced_frame = ttk.LabelFrame(self.frame, text="Advanced Settings")
        advanced_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=10)
        
        # Max tokens
        ttk.Label(advanced_frame, text="Max Tokens:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.max_tokens_var = tk.IntVar(value=config.get("llm_max_tokens", 150))
        self.max_tokens_entry = ttk.Entry(advanced_frame, textvariable=self.max_tokens_var, width=10)
        self.max_tokens_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Temperature
        ttk.Label(advanced_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.temperature_var = tk.DoubleVar(value=config.get("llm_temperature", 0.7))
        self.temperature_scale = ttk.Scale(advanced_frame, from_=0.1, to=1.0, 
                                         variable=self.temperature_var, orient=tk.HORIZONTAL, length=150)
        self.temperature_scale.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.temperature_label = ttk.Label(advanced_frame, text=f"{self.temperature_var.get():.1f}")
        self.temperature_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.temperature_scale.bind("<Motion>", self._update_temperature_label)
        
        # Use LLM threshold
        ttk.Label(advanced_frame, text="Use LLM Probability:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=config.get("use_llm_threshold", 0.5))
        self.threshold_scale = ttk.Scale(advanced_frame, from_=0.0, to=1.0, 
                                       variable=self.threshold_var, orient=tk.HORIZONTAL, length=150)
        self.threshold_scale.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.threshold_label = ttk.Label(advanced_frame, text=f"{self.threshold_var.get():.1f}")
        self.threshold_label.grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.threshold_scale.bind("<Motion>", self._update_threshold_label)
        
        # Azure endpoint (only visible for Azure)
        self.azure_frame = ttk.Frame(advanced_frame)
        self.azure_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        self.azure_frame.grid_remove()  # Hidden by default
        
        ttk.Label(self.azure_frame, text="Azure Endpoint:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.azure_endpoint_var = tk.StringVar(value=config.get("azure_endpoint", ""))
        self.azure_endpoint_entry = ttk.Entry(self.azure_frame, textvariable=self.azure_endpoint_var, width=40)
        self.azure_endpoint_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Local model path (only visible for local)
        self.local_frame = ttk.Frame(advanced_frame)
        self.local_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        self.local_frame.grid_remove()  # Hidden by default
        
        ttk.Label(self.local_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.local_path_var = tk.StringVar(value=config.get("local_model_path", ""))
        self.local_path_entry = ttk.Entry(self.local_frame, textvariable=self.local_path_var, width=40)
        self.local_path_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.browse_btn = ttk.Button(self.local_frame, text="Browse", command=self._browse_model_path)
        self.browse_btn.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Save button
        self.save_btn = ttk.Button(self.frame, text="Save LLM Settings", command=self._save_settings)
        self.save_btn.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Test button
        self.test_btn = ttk.Button(self.frame, text="Test LLM Connection", command=self._test_connection)
        self.test_btn.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Update visibility based on initial provider
        self._update_provider_specific_fields()
    
    def _get_current_model(self) -> str:
        """
        Get the current model based on the selected provider
        
        Returns:
            str: Current model name
        """
        provider = self.provider_var.get()
        return self.config.get(f"{provider}_model", self._get_default_model(provider))
    
    def _get_current_api_key(self) -> str:
        """
        Get the current API key based on the selected provider
        
        Returns:
            str: Current API key
        """
        provider = self.provider_var.get()
        env_key = os.environ.get(f"{provider.upper()}_API_KEY", "")
        return self.config.get(f"{provider}_api_key", env_key)
    
    def _get_default_model(self, provider: str) -> str:
        """
        Get the default model for a provider
        
        Args:
            provider: Provider name
        
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
        return defaults.get(provider, "gpt-3.5-turbo")
    
    def _update_model_options(self, event=None) -> None:
        """
        Update model dropdown options based on selected provider
        
        Args:
            event: ComboBox event (not used)
        """
        provider = self.provider_var.get()
        
        # Set model options based on provider
        if provider == "openai":
            models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
        elif provider in ["claude", "anthropic"]:
            models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        elif provider == "google":
            models = ["gemini-pro", "gemini-pro-vision"]
        elif provider == "azure":
            models = ["gpt-4", "gpt-35-turbo", "gpt-4-turbo"]
        elif provider == "mistral":
            models = ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large"]
        elif provider == "ollama":
            models = ["llama2", "mistral", "mixtral", "phi", "nous-hermes", "neural-chat", "codellama", "llava"]
        elif provider == "local":
            models = ["mistral-7b-instruct", "llama-7b-chat", "phi-2", "gemma-7b-it"]
        else:
            models = []
        
        # Update dropdown values
        self.model_dropdown['values'] = models
        
        # Set current model or default
        current_model = self.config.get(f"{provider}_model", self._get_default_model(provider))
        if current_model in models:
            self.model_var.set(current_model)
        elif models:
            self.model_var.set(models[0])
        
        # Update API key field
        self.api_key_var.set(self._get_current_api_key())
        
        # Update provider-specific fields
        self._update_provider_specific_fields()
    
    def _update_provider_specific_fields(self) -> None:
        """
        Show/hide provider-specific configuration fields
        """
        provider = self.provider_var.get()
        
        # Azure-specific fields
        if provider == "azure":
            self.azure_frame.grid()
        else:
            self.azure_frame.grid_remove()
        
        # Local model-specific fields
        if provider == "local":
            self.local_frame.grid()
        else:
            self.local_frame.grid_remove()
    
    def _toggle_key_visibility(self) -> None:
        """
        Toggle the visibility of the API key
        """
        if self.show_key.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")

    def _update_temperature_label(self, event=None) -> None:
        """
        Update temperature label when slider moves
        """
        self.temperature_label.config(text=f"{self.temperature_var.get():.1f}")
    
    def _update_threshold_label(self, event=None) -> None:
        """
        Update threshold label when slider moves
        """
        self.threshold_label.config(text=f"{self.threshold_var.get():.1f}")
    
    def _browse_model_path(self) -> None:
        """
        Open file browser to select local model path
        """
        from tkinter import filedialog
        
        directory = filedialog.askdirectory(
            title="Select directory containing the model files",
            initialdir=self.local_path_var.get() if self.local_path_var.get() else "/"
        )
        
        if directory:
            self.local_path_var.set(directory)
    
    def _save_settings(self) -> None:
        """
        Save LLM settings to configuration
        """
        provider = self.provider_var.get()
        model = self.model_var.get()
        api_key = self.api_key_var.get()
        
        # Update configuration
        self.config["llm_provider"] = provider
        self.config[f"{provider}_model"] = model
        self.config[f"{provider}_api_key"] = api_key
        self.config["llm_max_tokens"] = self.max_tokens_var.get()
        self.config["llm_temperature"] = self.temperature_var.get()
        self.config["use_llm_threshold"] = self.threshold_var.get()
        
        # Provider-specific settings
        if provider == "azure":
            self.config["azure_endpoint"] = self.azure_endpoint_var.get()
        
        if provider == "local":
            self.config["local_model_path"] = self.local_path_var.get()
        
        # Call save callback
        if self.save_callback:
            self.save_callback(self.config)
        
        messagebox.showinfo("Settings Saved", "LLM settings have been saved successfully!")
    
    def _test_connection(self) -> None:
        """
        Test the connection to the LLM provider
        """
        provider = self.provider_var.get()
        model = self.model_var.get()
        api_key = self.api_key_var.get()
        
        # Show testing message
        messagebox.showinfo("Testing Connection", 
                          f"Testing connection to {provider} with model {model}...\n\nThis may take a moment.")
        
        # Create a temporary config for testing
        test_config = {
            "llm_provider": provider,
            f"{provider}_model": model,
            f"{provider}_api_key": api_key,
            "llm_max_tokens": self.max_tokens_var.get(),
            "llm_temperature": self.temperature_var.get()
        }
        
        # Add provider-specific settings
        if provider == "azure":
            test_config["azure_endpoint"] = self.azure_endpoint_var.get()
        
        if provider == "local":
            test_config["local_model_path"] = self.local_path_var.get()
        
        # Import here to avoid circular imports
        from src.social.llm_interface import LLMInterface
        
        try:
            # Initialize LLM interface with test config
            llm = LLMInterface(test_config)
            
            # Try a simple test prompt
            response = llm.generate_chat_response(
                "Hello, how are you?", 
                "TestUser", 
                "whisper",
                {"current_activity": "testing LLM connection"}
            )
            
            # Show success message with response
            messagebox.showinfo("Connection Successful", 
                              f"Successfully connected to {provider}!\n\nResponse: {response}")
            
        except Exception as e:
            # Show error message
            messagebox.showerror("Connection Failed", 
                               f"Failed to connect to {provider}:\n\n{str(e)}\n\n"
                               f"Please check your API key and settings.")
            
class LLMUsagePanel:
    """
    Panel for configuring when to use LLM vs templated responses
    """
    
    def __init__(self, parent, config: Dict, save_callback: Callable):
        """
        Initialize the LLM usage configuration panel
        
        Args:
            parent: Parent widget
            config: Current configuration
            save_callback: Callback function to save configuration
        """
        self.parent = parent
        self.config = config
        self.save_callback = save_callback
        
        self.frame = ttk.LabelFrame(parent, text="LLM Usage Settings")
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Threshold controls
        threshold_frame = ttk.Frame(self.frame)
        threshold_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(threshold_frame, text="LLM Usage Probability:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=config.get("use_llm_threshold", 0.5))
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                                  variable=self.threshold_var, 
                                  orient=tk.HORIZONTAL, length=200)
        threshold_scale.grid(row=0, column=1, padx=5, pady=5)
        
        threshold_label = ttk.Label(threshold_frame, text=f"{self.threshold_var.get():.1f}")
        threshold_label.grid(row=0, column=2, padx=5, pady=5)
        
        def update_threshold_label(event):
            threshold_label.config(text=f"{self.threshold_var.get():.1f}")
        
        threshold_scale.bind("<Motion>", update_threshold_label)
        
        # Channel priority
        ttk.Label(self.frame, text="Always use LLM for these channels:").pack(anchor="w", padx=10, pady=(10, 5))
        
        channels_frame = ttk.Frame(self.frame)
        channels_frame.pack(fill="x", padx=10, pady=5)
        
        # Create variables for each channel
        self.channel_vars = {}
        default_channels = config.get("prioritize_llm_channels", ["whisper", "party", "guild"])
        
        for i, channel in enumerate(["whisper", "say", "party", "guild", "raid"]):
            var = tk.BooleanVar(value=channel in default_channels)
            self.channel_vars[channel] = var
            
            cb = ttk.Checkbutton(channels_frame, text=channel.capitalize(), variable=var)
            cb.grid(row=i//3, column=i%3, sticky="w", padx=5, pady=2)
        
        # Feature toggles
        ttk.Label(self.frame, text="Use LLM for:").pack(anchor="w", padx=10, pady=(15, 5))
        
        features_frame = ttk.Frame(self.frame)
        features_frame.pack(fill="x", padx=10, pady=5)
        
        # Create variables for each feature
        self.feature_vars = {}
        features = [
            ("group_chat", "Group Chat", True),
            ("emotes", "Emote Responses", False),
            ("quest_help", "Quest Help", True),
            ("vendor_dialog", "Vendor Dialog", False),
            ("combat_calls", "Combat Calls", False)
        ]
        
        for i, (feature_id, feature_name, default_value) in enumerate(features):
            var = tk.BooleanVar(value=config.get(f"use_llm_for_{feature_id}", default_value))
            self.feature_vars[feature_id] = var
            
            cb = ttk.Checkbutton(features_frame, text=feature_name, variable=var)
            cb.grid(row=i//2, column=i%2, sticky="w", padx=5, pady=2)
        
        # Token budget settings
        ttk.Label(self.frame, text="Token Budget:").pack(anchor="w", padx=10, pady=(15, 5))
        
        token_frame = ttk.Frame(self.frame)
        token_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(token_frame, text="Daily limit:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.daily_token_var = tk.IntVar(value=config.get("daily_token_limit", 10000))
        daily_token_entry = ttk.Entry(token_frame, textvariable=self.daily_token_var, width=10)
        daily_token_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Save button
        save_btn = ttk.Button(self.frame, text="Save Usage Settings", command=self._save_settings)
        save_btn.pack(pady=10)
    
    def _save_settings(self) -> None:
        """
        Save LLM usage settings
        """
        # Update threshold
        self.config["use_llm_threshold"] = self.threshold_var.get()
        
        # Update channel priorities
        prioritize_channels = [channel for channel, var in self.channel_vars.items() if var.get()]
        self.config["prioritize_llm_channels"] = prioritize_channels
        
        # Update feature toggles
        for feature_id, var in self.feature_vars.items():
            self.config[f"use_llm_for_{feature_id}"] = var.get()
        
        # Update token budget
        self.config["daily_token_limit"] = self.daily_token_var.get()
        
        # Call save callback
        if self.save_callback:
            self.save_callback(self.config)
        
        tk.messagebox.showinfo("Settings Saved", "LLM usage settings have been saved!")