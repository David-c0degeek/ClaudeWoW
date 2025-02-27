# src/utils/gui_overlay.py

import tkinter as tk
from tkinter import ttk
import threading
import time
import logging
import queue
from typing import Dict, List, Any, Optional

class GUIOverlay:
    """
    GUI overlay for monitoring and controlling the WoW AI player
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GUI overlay
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.utils.gui_overlay")
        self.config = config
        
        # Create a queue for thread-safe updates
        self.update_queue = queue.Queue()
        
        # Create window
        self.root = tk.Tk()
        self.root.title("WoW AI Player Monitor")
        self.root.geometry("700x500")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Set opacity if configured
        self.opacity = config.get("overlay_opacity", 0.9)
        self.root.attributes("-alpha", self.opacity)
        
        # Create the interface
        self._create_interface()
        
        # State variables
        self.running = True
        self.paused = False
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("GUI overlay initialized")
    
    def _create_interface(self) -> None:
        """
        Create the GUI interface components
        """
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.pack(fill="x", padx=5, pady=5)
        
        # Status indicators
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 12, "bold"))
        self.status_label.pack(side="left", padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(status_frame)
        control_frame.pack(side="right", padx=5)
        
        self.pause_btn = ttk.Button(control_frame, text="Pause", command=self._toggle_pause)
        self.pause_btn.pack(side="left", padx=2)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._stop)
        self.stop_btn.pack(side="left", padx=2)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Perception tab
        perception_tab = ttk.Frame(self.notebook)
        self.notebook.add(perception_tab, text="Perception")
        
        ttk.Label(perception_tab, text="Game State:").pack(anchor="w", padx=5, pady=5)
        
        game_state_frame = ttk.Frame(perception_tab)
        game_state_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Game state display
        self.game_state_text = tk.Text(game_state_frame, height=15, width=80, wrap="word")
        self.game_state_text.pack(side="left", fill="both", expand=True)
        
        game_state_scroll = ttk.Scrollbar(game_state_frame, command=self.game_state_text.yview)
        game_state_scroll.pack(side="right", fill="y")
        self.game_state_text.config(yscrollcommand=game_state_scroll.set)
        
        # Decision tab
        decision_tab = ttk.Frame(self.notebook)
        self.notebook.add(decision_tab, text="Decisions")
        
        ttk.Label(decision_tab, text="Current Goal:").pack(anchor="w", padx=5, pady=5)
        
        self.current_goal_var = tk.StringVar(value="No active goal")
        ttk.Label(decision_tab, textvariable=self.current_goal_var, font=("Arial", 10, "bold")).pack(anchor="w", padx=5)
        
        ttk.Label(decision_tab, text="Plan Steps:").pack(anchor="w", padx=5, pady=5)
        
        plan_frame = ttk.Frame(decision_tab)
        plan_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.plan_text = tk.Text(plan_frame, height=10, width=80, wrap="word")
        self.plan_text.pack(side="left", fill="both", expand=True)
        
        plan_scroll = ttk.Scrollbar(plan_frame, command=self.plan_text.yview)
        plan_scroll.pack(side="right", fill="y")
        self.plan_text.config(yscrollcommand=plan_scroll.set)
        
        ttk.Label(decision_tab, text="Recent Actions:").pack(anchor="w", padx=5, pady=5)
        
        actions_frame = ttk.Frame(decision_tab)
        actions_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.actions_text = tk.Text(actions_frame, height=10, width=80, wrap="word")
        self.actions_text.pack(side="left", fill="both", expand=True)
        
        actions_scroll = ttk.Scrollbar(actions_frame, command=self.actions_text.yview)
        actions_scroll.pack(side="right", fill="y")
        self.actions_text.config(yscrollcommand=actions_scroll.set)
        
        # Social tab
        social_tab = ttk.Frame(self.notebook)
        self.notebook.add(social_tab, text="Social")
        
        ttk.Label(social_tab, text="Recent Chat:").pack(anchor="w", padx=5, pady=5)
        
        chat_frame = ttk.Frame(social_tab)
        chat_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.chat_text = tk.Text(chat_frame, height=15, width=80, wrap="word")
        self.chat_text.pack(side="left", fill="both", expand=True)
        
        chat_scroll = ttk.Scrollbar(chat_frame, command=self.chat_text.yview)
        chat_scroll.pack(side="right", fill="y")
        self.chat_text.config(yscrollcommand=chat_scroll.set)
        
        # Manual message frame
        manual_frame = ttk.LabelFrame(social_tab, text="Send Manual Message")
        manual_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(manual_frame, text="Channel:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.channel_var = tk.StringVar(value="say")
        channel_options = ["say", "party", "guild", "raid", "whisper"]
        channel_dropdown = ttk.Combobox(manual_frame, textvariable=self.channel_var, values=channel_options, width=10)
        channel_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.whisper_target_var = tk.StringVar()
        self.whisper_target_entry = ttk.Entry(manual_frame, textvariable=self.whisper_target_var, width=15)
        self.whisper_target_entry.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.whisper_target_entry.grid_remove()  # Hidden initially
        
        def on_channel_change(*args):
            if self.channel_var.get() == "whisper":
                ttk.Label(manual_frame, text="Target:").grid(row=0, column=2, padx=(10,0), pady=5, sticky="e")
                self.whisper_target_entry.grid()
            else:
                self.whisper_target_entry.grid_remove()
        
        self.channel_var.trace("w", on_channel_change)
        
        ttk.Label(manual_frame, text="Message:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        message_frame = ttk.Frame(manual_frame)
        message_frame.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        
        self.message_var = tk.StringVar()
        message_entry = ttk.Entry(message_frame, textvariable=self.message_var, width=50)
        message_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        send_btn = ttk.Button(message_frame, text="Send", command=self._send_manual_message)
        send_btn.pack(side="right", padx=5)
        
        # Logs tab
        logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(logs_tab, text="Logs")
        
        log_frame = ttk.Frame(logs_tab)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=20, width=80, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)
        
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=log_scroll.set)
        
        # Create log handler to redirect logs to the GUI
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.insert("end", msg + "\n")
                self.text_widget.see("end")
        
        # Create and add handler to root logger
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
        
        # Status bar
        status_bar = ttk.Frame(main_frame)
        status_bar.pack(fill="x", padx=5, pady=5)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = ttk.Label(status_bar, textvariable=self.fps_var)
        fps_label.pack(side="right")
        
        self.time_var = tk.StringVar(value="Runtime: 0:00:00")
        time_label = ttk.Label(status_bar, textvariable=self.time_var)
        time_label.pack(side="left")
        
        # Add token usage tracking
        self.token_var = tk.StringVar(value="Tokens: 0")
        token_label = ttk.Label(status_bar, textvariable=self.token_var)
        token_label.pack(side="right", padx=10)
    
    def start(self) -> None:
        """
        Start the GUI loop
        """
        self.start_time = time.time()
        self.root.after(100, self._update_runtime)
        self.root.mainloop()
    
    def update_state(self, game_state: Any, agent_state: Optional[Dict] = None, 
                    social_state: Optional[Dict] = None) -> None:
        """
        Update the displayed state
        
        Args:
            game_state: Current game state
            agent_state: Agent state information
            social_state: Social state information
        """
        # Put the update in the queue for thread-safe processing
        self.update_queue.put({
            "type": "state_update",
            "game_state": game_state,
            "agent_state": agent_state,
            "social_state": social_state,
            "timestamp": time.time()
        })
    
    def update_actions(self, actions: List[Dict]) -> None:
        """
        Update the actions display
        
        Args:
            actions: List of recent actions
        """
        self.update_queue.put({
            "type": "actions_update",
            "actions": actions,
            "timestamp": time.time()
        })
    
    def update_chat(self, chat_entry: Dict) -> None:
        """
        Update the chat display
        
        Args:
            chat_entry: Chat entry to add
        """
        self.update_queue.put({
            "type": "chat_update",
            "chat_entry": chat_entry,
            "timestamp": time.time()
        })
    
    def update_log(self, log_entry: str) -> None:
        """
        Add a log entry
        
        Args:
            log_entry: Log entry to add
        """
        self.update_queue.put({
            "type": "log_update",
            "log_entry": log_entry,
            "timestamp": time.time()
        })
    
    def _process_queue(self) -> None:
        """
        Process the update queue in the background
        """
        last_fps_update = time.time()
        frame_count = 0
        
        while self.running:
            # Process all available updates
            updates_processed = 0
            
            while not self.update_queue.empty() and updates_processed < 100:
                try:
                    update = self.update_queue.get_nowait()
                    
                    if update["type"] == "state_update":
                        self._update_state_display(update["game_state"], 
                                                 update["agent_state"], 
                                                 update["social_state"])
                    
                    elif update["type"] == "actions_update":
                        self._update_actions_display(update["actions"])
                    
                    elif update["type"] == "chat_update":
                        self._update_chat_display(update["chat_entry"])
                    
                    elif update["type"] == "log_update":
                        self._update_log_display(update["log_entry"])
                    
                    self.update_queue.task_done()
                    updates_processed += 1
                    frame_count += 1
                except queue.Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing update: {e}")
            
            # Update FPS counter approximately once per second
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                fps = frame_count / (current_time - last_fps_update)
                self.fps_var.set(f"FPS: {fps:.1f}")
                last_fps_update = current_time
                frame_count = 0
            
            # Slight pause to avoid CPU hogging
            time.sleep(0.01)
    
    def _update_state_display(self, game_state: Any, agent_state: Optional[Dict], 
                            social_state: Optional[Dict]) -> None:
        """
        Update the state display with latest information
        
        Args:
            game_state: Current game state
            agent_state: Agent state information
            social_state: Social state information
        """
        # Update game state text
        if game_state:
            try:
                self.game_state_text.delete(1.0, tk.END)
                
                # Format game state properties
                state_text = ""
                for attr, value in sorted(game_state.__dict__.items()):
                    if attr.startswith("_"):
                        continue
                    
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        state_text += f"{attr}: [Complex data]\n"
                    else:
                        state_text += f"{attr}: {value}\n"
                
                self.game_state_text.insert(tk.END, state_text)
            except Exception as e:
                self.logger.error(f"Error updating game state display: {e}")
        
        # Update agent state (goal and plan)
        if agent_state:
            try:
                if "current_goal" in agent_state:
                    goal = agent_state["current_goal"]
                    if goal:
                        goal_type = goal.get("type", "unknown")
                        goal_desc = f"Goal: {goal_type.capitalize()}"
                        
                        if "target" in goal:
                            goal_desc += f" - Target: {goal['target']}"
                        
                        if "priority" in goal:
                            goal_desc += f" (Priority: {goal['priority']:.1f})"
                        
                        self.current_goal_var.set(goal_desc)
                    else:
                        self.current_goal_var.set("No active goal")
                
                if "current_plan" in agent_state:
                    plan = agent_state["current_plan"]
                    current_step = agent_state.get("current_plan_step", 0)
                    
                    self.plan_text.delete(1.0, tk.END)
                    
                    if plan:
                        for i, step in enumerate(plan):
                            prefix = "→ " if i == current_step else "  "
                            step_desc = step.get("description", step.get("type", "Unknown step"))
                            self.plan_text.insert(tk.END, f"{prefix}{i+1}. {step_desc}\n")
                    else:
                        self.plan_text.insert(tk.END, "No active plan")
            except Exception as e:
                self.logger.error(f"Error updating agent state display: {e}")
        
        # Update social state display
        if social_state:
            try:
                # Social state would be incorporated here
                pass
            except Exception as e:
                self.logger.error(f"Error updating social state display: {e}")
        
        # Update status indicator
        if game_state:
            status = "Online"
            
            if self.paused:
                status = "Paused"
            
            if hasattr(game_state, "is_in_combat") and game_state.is_in_combat:
                status += " - In Combat"
            
            if hasattr(game_state, "current_zone") and game_state.current_zone:
                status += f" - {game_state.current_zone}"
            
            self.status_var.set(status)
    
    def _update_actions_display(self, actions: List[Dict]) -> None:
        """
        Update the actions display
        
        Args:
            actions: List of recent actions
        """
        try:
            # Display the actions in the text widget
            self.actions_text.delete(1.0, tk.END)
            
            if actions:
                for i, action in enumerate(actions):
                    action_desc = action.get("description", "")
                    action_type = action.get("type", "unknown")
                    
                    if not action_desc:
                        # Generate description from action properties
                        if action_type == "move":
                            action_desc = f"Move to ({action.get('x', 0)}, {action.get('y', 0)})"
                        elif action_type == "cast":
                            action_desc = f"Cast {action.get('spell', 'unknown')} on {action.get('target', 'unknown')}"
                        elif action_type == "interact":
                            action_desc = f"Interact with {action.get('target', 'unknown')}"
                        else:
                            action_desc = f"{action_type.capitalize()} action"
                    
                    self.actions_text.insert(tk.END, f"{i+1}. {action_desc}\n")
            else:
                self.actions_text.insert(tk.END, "No recent actions")
        except Exception as e:
            self.logger.error(f"Error updating actions display: {e}")
    
    def _update_chat_display(self, chat_entry: Dict) -> None:
        """
        Update the chat display
        
        Args:
            chat_entry: Chat entry to add
        """
        try:
            # Format chat entry based on type
            sender = chat_entry.get("sender", "Unknown")
            message = chat_entry.get("message", "")
            channel = chat_entry.get("channel", "say")
            
            timestamp = chat_entry.get("timestamp", time.time())
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            # Format based on channel
            if channel == "whisper":
                target = chat_entry.get("target", "")
                if sender == "self":
                    formatted = f"[{time_str}] To {target}: {message}\n"
                else:
                    formatted = f"[{time_str}] {sender} whispers: {message}\n"
            else:
                channel_prefix = channel.capitalize()
                if sender == "self":
                    formatted = f"[{time_str}] [{channel_prefix}]: {message}\n"
                else:
                    formatted = f"[{time_str}] [{channel_prefix}] {sender}: {message}\n"
            
            # Add to chat text
            self.chat_text.insert(tk.END, formatted)
            self.chat_text.see(tk.END)
            
            # Limit chat history (remove oldest entries if too many)
            content = self.chat_text.get(1.0, tk.END)
            lines = content.split("\n")
            if len(lines) > 200:  # Limit to 200 lines
                self.chat_text.delete(1.0, f"{len(lines) - 200}.0")
        except Exception as e:
            self.logger.error(f"Error updating chat display: {e}")
    
    def _update_log_display(self, log_entry: str) -> None:
        """
        Update the log display
        
        Args:
            log_entry: Log entry to add
        """
        try:
            self.log_text.insert(tk.END, log_entry + "\n")
            self.log_text.see(tk.END)
        except Exception as e:
            self.logger.error(f"Error updating log display: {e}")
    
    def _update_runtime(self) -> None:
        """
        Update the runtime display
        """
        if not self.running:
            return
        
        try:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            time_str = f"Runtime: {hours}:{minutes:02d}:{seconds:02d}"
            self.time_var.set(time_str)
        except Exception as e:
            self.logger.error(f"Error updating runtime: {e}")
        
        # Schedule next update
        self.root.after(1000, self._update_runtime)
    
    def _toggle_pause(self) -> None:
        """
        Toggle pause state
        """
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
    
    def _stop(self) -> None:
        """
        Stop the AI
        """
        if tk.messagebox.askyesno("Confirm", "Are you sure you want to stop the AI player?"):
            self.running = False
            self.root.quit()
    
    def _send_manual_message(self) -> None:
        """
        Send a manual chat message
        """
        channel = self.channel_var.get()
        message = self.message_var.get()
        
        if not message:
            return
        
        # Create chat entry
        chat_entry = {
            "sender": "self",
            "message": message,
            "channel": channel,
            "timestamp": time.time()
        }
        
        # Add target for whispers
        if channel == "whisper":
            target = self.whisper_target_var.get()
            if not target:
                tk.messagebox.showerror("Error", "Please enter a whisper target")
                return
            chat_entry["target"] = target
        
        # Add to chat display
        self._update_chat_display(chat_entry)
        
        # Clear message field
        self.message_var.set("")
        
        # Trigger chat in the AI system (via callback)
        if hasattr(self, "chat_callback") and self.chat_callback:
            self.chat_callback(chat_entry)
    
    def set_chat_callback(self, callback: callable) -> None:
        """
        Set the callback for when a manual chat is sent
        
        Args:
            callback: Function to call with chat data
        """
        self.chat_callback = callback
    
    def is_paused(self) -> bool:
        """
        Check if the AI is paused
        
        Returns:
            bool: True if paused
        """
        return self.paused
    
    def _on_close(self) -> None:
        """
        Handle window close event
        """
        if tk.messagebox.askyesno("Confirm", "Close the GUI overlay? The AI will continue running."):
            self.running = False
            self.root.destroy()

    def update_token_usage(self, tokens: int) -> None:
        """
        Update token usage display

        Args:
            tokens: Total tokens used
        """
        try:
            self.token_var.set(f"Tokens: {tokens:,}")
        except Exception as e:
            self.logger.error(f"Error updating token usage: {e}")