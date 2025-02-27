"""
WoW AI Player Launcher

This script initializes and starts the WoW AI player.
"""

import os
import sys
import logging
import time
import json
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config, save_config
from src.utils.llm_config import LLMConfigPanel
from src.perception.screen_reader import ScreenReader
from src.action.controller import Controller
from src.decision.agent import Agent

def setup_logging(log_level):
    """Configure logging"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"wow_ai_{timestamp}.log")
    
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("wow_ai")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="WoW AI Player")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("--delay", type=float, default=0,
                        help="Delay in seconds before starting (to give time to switch to the game window)")
    
    parser.add_argument("--gui", action="store_true", help="Launch in GUI configuration mode")
    
    return parser.parse_args()

def launch_gui(config_path=None):
    """
    Launch the GUI configuration tool
    
    Args:
        config_path: Path to config file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create main window
    root = tk.Tk()
    root.title("WoW AI Player Configuration")
    root.geometry("800x600")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # General settings tab
    general_tab = ttk.Frame(notebook)
    notebook.add(general_tab, text="General Settings")
    
    # Game settings
    game_frame = ttk.LabelFrame(general_tab, text="Game Settings")
    game_frame.pack(padx=10, pady=10, fill="both")
    
    ttk.Label(game_frame, text="Game Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    game_path_var = tk.StringVar(value=config.get("game_path", ""))
    game_path_entry = ttk.Entry(game_frame, textvariable=game_path_var, width=50)
    game_path_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    
    def browse_game_path():
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select World of Warcraft executable",
            filetypes=(("Executable files", "*.exe"), ("All files", "*.*"))
        )
        if file_path:
            game_path_var.set(file_path)
    
    browse_button = ttk.Button(game_frame, text="Browse", command=browse_game_path)
    browse_button.grid(row=0, column=2, padx=5, pady=5)
    
    # Resolution settings
    ttk.Label(game_frame, text="Resolution:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    
    resolution_frame = ttk.Frame(game_frame)
    resolution_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
    
    width_var = tk.IntVar(value=config.get("resolution", {}).get("width", 1920))
    height_var = tk.IntVar(value=config.get("resolution", {}).get("height", 1080))
    
    ttk.Label(resolution_frame, text="Width:").pack(side=tk.LEFT)
    width_entry = ttk.Entry(resolution_frame, textvariable=width_var, width=5)
    width_entry.pack(side=tk.LEFT, padx=2)
    
    ttk.Label(resolution_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
    height_entry = ttk.Entry(resolution_frame, textvariable=height_var, width=5)
    height_entry.pack(side=tk.LEFT, padx=2)
    
    # UI Scale
    ttk.Label(game_frame, text="UI Scale:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    
    ui_scale_var = tk.DoubleVar(value=config.get("ui_scale", 1.0))
    ui_scale_entry = ttk.Entry(game_frame, textvariable=ui_scale_var, width=5)
    ui_scale_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
    
    # Player information
    player_frame = ttk.LabelFrame(general_tab, text="Player Character")
    player_frame.pack(padx=10, pady=10, fill="both")
    
    ttk.Label(player_frame, text="Character Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    player_name_var = tk.StringVar(value=config.get("player_name", ""))
    player_name_entry = ttk.Entry(player_frame, textvariable=player_name_var, width=20)
    player_name_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    
    ttk.Label(player_frame, text="Class:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    player_class_var = tk.StringVar(value=config.get("player_class", ""))
    player_classes = ["Warrior", "Paladin", "Hunter", "Rogue", "Priest", 
                     "Shaman", "Mage", "Warlock", "Druid", "Death Knight"]
    player_class_dropdown = ttk.Combobox(player_frame, textvariable=player_class_var, values=player_classes)
    player_class_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
    
    ttk.Label(player_frame, text="Race:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    player_race_var = tk.StringVar(value=config.get("player_race", ""))
    player_races = ["Human", "Dwarf", "Night Elf", "Gnome", "Draenei", "Worgen",
                   "Orc", "Undead", "Tauren", "Troll", "Blood Elf", "Goblin"]
    player_race_dropdown = ttk.Combobox(player_frame, textvariable=player_race_var, values=player_races)
    player_race_dropdown.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
    
    ttk.Label(player_frame, text="Personality:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    player_personality_var = tk.StringVar(value=config.get("player_personality", "friendly and helpful"))
    player_personality_entry = ttk.Entry(player_frame, textvariable=player_personality_var, width=40)
    player_personality_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
    
    # LLM settings tab
    llm_tab = ttk.Frame(notebook)
    notebook.add(llm_tab, text="LLM Settings")

    # Create notebook for LLM settings subtabs
    llm_notebook = ttk.Notebook(llm_tab)
    llm_notebook.pack(fill="both", expand=True, padx=5, pady=5)

    # Provider settings tab
    provider_tab = ttk.Frame(llm_notebook)
    llm_notebook.add(provider_tab, text="Provider")

    # Initialize LLM config panel
    def save_llm_config(updated_config):
        # Update only LLM-related settings
        for key, value in updated_config.items():
            if key.startswith("llm_") or "_api_key" in key or "_model" in key:
                config[key] = value

    llm_config_panel = LLMConfigPanel(provider_tab, config, save_llm_config)

    # Usage settings tab
    usage_tab = ttk.Frame(llm_notebook)
    llm_notebook.add(usage_tab, text="Usage Settings")

    # Initialize LLM usage panel
    def save_usage_config(updated_config):
        # Update usage-related settings
        for key, value in updated_config.items():
            if key.startswith("use_llm_") or key in ["prioritize_llm_channels", "daily_token_limit"]:
                config[key] = value

    llm_usage_panel = LLMUsagePanel(usage_tab, config, save_usage_config)
    
    # Social settings tab
    social_tab = ttk.Frame(notebook)
    notebook.add(social_tab, text="Social Settings")
    
    social_frame = ttk.LabelFrame(social_tab, text="Social Behavior")
    social_frame.pack(padx=10, pady=10, fill="both")
    
    # Friendliness
    ttk.Label(social_frame, text="Friendliness:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    
    friendliness_var = tk.DoubleVar(value=config.get("social_friendliness", 0.5))
    friendliness_scale = ttk.Scale(social_frame, from_=0.0, to=1.0, 
                                  variable=friendliness_var, orient=tk.HORIZONTAL, length=200)
    friendliness_scale.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    
    friendliness_label = ttk.Label(social_frame, text=f"{friendliness_var.get():.1f}")
    friendliness_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
    
    def update_friendliness_label(event):
        friendliness_label.config(text=f"{friendliness_var.get():.1f}")
    
    friendliness_scale.bind("<Motion>", update_friendliness_label)
    
    # Chattiness
    ttk.Label(social_frame, text="Chattiness:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    
    chattiness_var = tk.DoubleVar(value=config.get("social_chattiness", 0.5))
    chattiness_scale = ttk.Scale(social_frame, from_=0.0, to=1.0, 
                                variable=chattiness_var, orient=tk.HORIZONTAL, length=200)
    chattiness_scale.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
    
    chattiness_label = ttk.Label(social_frame, text=f"{chattiness_var.get():.1f}")
    chattiness_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
    
    def update_chattiness_label(event):
        chattiness_label.config(text=f"{chattiness_var.get():.1f}")
    
    chattiness_scale.bind("<Motion>", update_chattiness_label)
    
    # Helpfulness
    ttk.Label(social_frame, text="Helpfulness:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    
    helpfulness_var = tk.DoubleVar(value=config.get("social_helpfulness", 0.8))
    helpfulness_scale = ttk.Scale(social_frame, from_=0.0, to=1.0, 
                                 variable=helpfulness_var, orient=tk.HORIZONTAL, length=200)
    helpfulness_scale.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
    
    helpfulness_label = ttk.Label(social_frame, text=f"{helpfulness_var.get():.1f}")
    helpfulness_label.grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
    
    def update_helpfulness_label(event):
        helpfulness_label.config(text=f"{helpfulness_var.get():.1f}")
    
    helpfulness_scale.bind("<Motion>", update_helpfulness_label)
    
    # Chat settings
    chat_frame = ttk.LabelFrame(social_tab, text="Chat Settings")
    chat_frame.pack(padx=10, pady=10, fill="both")
    
    ttk.Label(chat_frame, text="Max messages per minute:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    
    max_msgs_var = tk.IntVar(value=config.get("max_messages_per_minute", 8))
    max_msgs_entry = ttk.Entry(chat_frame, textvariable=max_msgs_var, width=5)
    max_msgs_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    
    ttk.Label(chat_frame, text="Min chat interval (seconds):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    
    min_chat_interval_var = tk.DoubleVar(value=config.get("min_chat_interval", 5.0))
    min_chat_interval_entry = ttk.Entry(chat_frame, textvariable=min_chat_interval_var, width=5)
    min_chat_interval_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
    
    # Actions tab
    actions_tab = ttk.Frame(notebook)
    notebook.add(actions_tab, text="Actions")
    
    # Launch button
    def save_and_launch():
        # Update config with GUI values
        config["game_path"] = game_path_var.get()
        config["resolution"] = {
            "width": width_var.get(),
            "height": height_var.get()
        }
        config["ui_scale"] = ui_scale_var.get()
        config["player_name"] = player_name_var.get()
        config["player_class"] = player_class_var.get()
        config["player_race"] = player_race_var.get()
        config["player_personality"] = player_personality_var.get()
        config["social_friendliness"] = friendliness_var.get()
        config["social_chattiness"] = chattiness_var.get()
        config["social_helpfulness"] = helpfulness_var.get()
        config["max_messages_per_minute"] = max_msgs_var.get()
        config["min_chat_interval"] = min_chat_interval_var.get()
        
        # Save config
        save_config(config, config_path)
        
        # Ask if user wants to launch now
        if messagebox.askyesno("Launch AI Player", "Configuration saved. Launch AI player now?"):
            root.destroy()
            main(config=config)
        else:
            messagebox.showinfo("Configuration Saved", "Configuration has been saved. You can launch the AI player later.")
    
    launch_frame = ttk.Frame(actions_tab)
    launch_frame.pack(padx=10, pady=10, fill="both", expand=True)
    
    save_button = ttk.Button(launch_frame, text="Save Configuration", command=lambda: save_config(config, config_path))
    save_button.pack(pady=5)
    
    launch_button = ttk.Button(launch_frame, text="Save and Launch AI Player", command=save_and_launch)
    launch_button.pack(pady=10)
    
    delay_var = tk.IntVar(value=5)
    
    ttk.Label(launch_frame, text="Startup delay (seconds):").pack(pady=5)
    delay_spinner = ttk.Spinbox(launch_frame, from_=0, to=30, textvariable=delay_var, width=5)
    delay_spinner.pack(pady=5)
    
    # Run the GUI
    root.mainloop()

# Update to launcher.py

def main(config=None):
    """Main entry point for the WoW AI Player"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Launch GUI if requested
    if args.gui:
        launch_gui(args.config)
        return
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting WoW AI Player")
    
    # Load configuration if not provided
    if config is None:
        config = load_config(args.config)
    logger.info("Configuration loaded")
    
    # Check if GUI overlay is enabled
    use_gui_overlay = config.get("use_gui_overlay", True)
    gui_overlay = None
    
    if use_gui_overlay:
        from src.utils.gui_overlay import GUIOverlay
        gui_overlay = GUIOverlay(config)
        logger.info("GUI overlay initialized")
    
    # Add delay if specified
    delay = args.delay if hasattr(args, 'delay') else config.get("startup_delay", 0)
    if delay > 0:
        logger.info(f"Waiting {delay} seconds before starting...")
        time.sleep(delay)
    
    try:
        # Initialize components
        screen_reader = ScreenReader(config)
        controller = Controller(config)
        agent = Agent(config, screen_reader, controller)
        
        logger.info("All components initialized successfully")
        
        # Set up GUI overlay callback for chat
        if gui_overlay:
            def handle_manual_chat(chat_entry):
                # Create a manual chat action
                channel = chat_entry.get("channel")
                message = chat_entry.get("message")
                
                action = {
                    "type": "chat",
                    "message": message,
                    "channel": channel,
                    "description": f"Manual chat: {message}"
                }
                
                # If it's a whisper, add target
                if channel == "whisper":
                    action["target"] = chat_entry.get("target")
                
                # Execute the chat action
                controller.execute([action])
            
            gui_overlay.set_chat_callback(handle_manual_chat)
            
            # Start GUI in a separate thread
            gui_thread = threading.Thread(target=gui_overlay.start, daemon=True)
            gui_thread.start()
        
        # Main loop
        logger.info("Entering main loop")
        
        cycle_counter = 0
        fps_counter = 0
        fps_timer = time.time()
        
        while True:
            cycle_counter += 1
            fps_counter += 1
            cycle_start_time = time.time()
            
            # Check if paused via GUI
            if gui_overlay and gui_overlay.is_paused():
                time.sleep(0.1)  # Reduce CPU usage while paused
                continue
            
            try:
                # Process game state
                game_state = screen_reader.capture_game_state()
                
                # Decide on actions
                actions = agent.decide(game_state)
                
                # Update GUI if enabled
                if gui_overlay:
                    # Update state information
                    agent_state = {
                        "current_goal": agent.current_goal,
                        "current_plan": agent.current_plan,
                        "current_plan_step": agent.current_plan_step
                    }
                    
                    social_state = None
                    if hasattr(agent, "social_manager"):
                        social_state = {
                            "chat_history": agent.social_manager.chat_history,
                            "nearby_players": agent.social_manager.social_state.get("nearby_players", []),
                            "grouped_players": agent.social_manager.social_state.get("grouped_players", [])
                        }
                    
                    gui_overlay.update_state(game_state, agent_state, social_state)
                    gui_overlay.update_actions(actions)
                
                # Execute actions if not empty
                if actions:
                    action_descriptions = [a.get("description", a.get("type", "unknown")) for a in actions]
                    logger.info(f"Executing actions: {', '.join(action_descriptions)}")
                    controller.execute(actions)
                    
                    # Update GUI with chat actions
                    if gui_overlay:
                        for action in actions:
                            if action.get("type") == "chat":
                                chat_entry = {
                                    "sender": "self",
                                    "message": action.get("message", ""),
                                    "channel": action.get("channel", "say"),
                                    "timestamp": time.time()
                                }
                                
                                if "target" in action:
                                    chat_entry["target"] = action["target"]
                                
                                gui_overlay.update_chat(chat_entry)
                
                # Calculate cycle time
                cycle_time = time.time() - cycle_start_time
                
                # Calculate FPS approximately once per second
                if time.time() - fps_timer >= 1.0:
                    fps = fps_counter / (time.time() - fps_timer)
                    logger.debug(f"FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_timer = time.time()
                
                # Sleep to maintain target cycle time
                target_cycle_time = config.get("loop_interval", 0.1)
                if cycle_time < target_cycle_time:
                    time.sleep(target_cycle_time - cycle_time)
                
                # Log performance every 100 cycles
                if cycle_counter % 100 == 0:
                    logger.info(f"Performance: {cycle_time:.3f}s per cycle, "
                               f"target: {target_cycle_time:.3f}s")
            
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.exception(f"Error in main loop cycle {cycle_counter}: {e}")
                # Brief pause to prevent error spam
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
    finally:
        # Stop GUI if running
        if gui_overlay:
            gui_overlay.running = False
        
        logger.info("WoW AI Player shutting down")

if __name__ == "__main__":
    main()