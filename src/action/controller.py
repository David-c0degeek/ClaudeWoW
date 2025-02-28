"""
Controller Module

This module handles action execution by converting decisions into game inputs.
"""

import logging
import time
import random
import math
from typing import Dict, List, Tuple, Any, Optional
import pyautogui
import pynput
from pynput import keyboard, mouse

class Controller:
    """
    Executes actions by controlling keyboard and mouse inputs
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Controller
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.action.controller")
        self.config = config
        
        # Input settings
        self.input_delay = config.get("input_delay", 0.05)  # seconds between inputs
        self.mouse_move_speed = config.get("mouse_move_speed", 0.1)  # seconds for mouse movement
        self.key_press_duration = config.get("key_press_duration", 0.05)  # seconds to hold keys
        
        # Initialize keyboard and mouse controllers
        self.keyboard = keyboard.Controller()
        self.mouse = mouse.Controller()
        
        # Store game window position and size
        self.window_position = (0, 0)  # Default to full screen
        self.window_size = (config["resolution"]["width"], config["resolution"]["height"])
        
        # Ability keybindings
        self.keybindings = self._load_keybindings()
        
        # Movement state
        self.is_moving_forward = False
        self.is_moving_backward = False
        self.is_strafing_left = False
        self.is_strafing_right = False
        self.is_turning_left = False
        self.is_turning_right = False
        
        # Action queue and timing
        self.last_action_time = 0
        self.action_queue = []
        self.current_action = None
        self.action_start_time = 0
        
        self.logger.info("Controller initialized")
    
    def _load_keybindings(self) -> Dict:
        """
        Load keybindings from configuration
        
        Returns:
            Dict: Keybinding dictionary
        """
        # Default keybindings
        default_bindings = {
            # Movement
            "move_forward": "w",
            "move_backward": "s",
            "strafe_left": "q",
            "strafe_right": "e",
            "turn_left": "a",
            "turn_right": "d",
            "jump": "space",
            "sit": "x",
            
            # Combat
            "auto_attack": "1",
            "target_nearest_enemy": "tab",
            "target_nearest_friend": "f",
            "interact": "right_click",
            "loot": "right_click",
            
            # Interface
            "open_map": "m",
            "open_bags": "b",
            "open_character": "c",
            "open_quest_log": "l",
            
            # Action bar slots
            "action_1": "1",
            "action_2": "2",
            "action_3": "3",
            "action_4": "4",
            "action_5": "5",
            "action_6": "6",
            "action_7": "7",
            "action_8": "8",
            "action_9": "9",
            "action_10": "0",
            "action_11": "-",
            "action_12": "=",
            
            # Modifier keys
            "shift": "shift",
            "ctrl": "ctrl",
            "alt": "alt"
        }
        
        # Load custom keybindings from config if present
        custom_bindings = self.config.get("keybindings", {})
        
        # Merge with default bindings
        bindings = {**default_bindings, **custom_bindings}
        
        return bindings
    
    def execute(self, actions: List[Dict]) -> None:
        """
        Execute a list of actions
        
        Args:
            actions: List of action dictionaries
        """
        if not actions:
            return
        
        try:
            # Add actions to queue
            for action in actions:
                self.action_queue.append(action)
            
            # Process action queue
            self._process_action_queue()
            
        except Exception as e:
            self.logger.error(f"Error executing actions: {e}")
            # Stop any ongoing movement
            self._stop_all_movement()
    
    def _process_action_queue(self) -> None:
        """
        Process the action queue
        """
        # Check if we're already processing an action
        current_time = time.time()
        
        if self.current_action:
            # Check if current action is completed
            action_duration = self.current_action.get("duration", 0)
            if current_time - self.action_start_time >= action_duration:
                # Action completed
                self._finish_current_action()
            else:
                # Action still in progress
                return
        
        # Check if we need to wait before next action
        if current_time - self.last_action_time < self.input_delay:
            return
        
        # Get next action from queue
        if not self.action_queue:
            return
        
        self.current_action = self.action_queue.pop(0)
        self.action_start_time = current_time
        
        # Execute the action
        action_type = self.current_action.get("type", "")
        
        if action_type == "move":
            self._execute_move(self.current_action)
        elif action_type == "turn":
            self._execute_turn(self.current_action)
        elif action_type == "cast":
            self._execute_cast(self.current_action)
        elif action_type == "target":
            self._execute_target(self.current_action)
        elif action_type == "interact":
            self._execute_interact(self.current_action)
        elif action_type == "loot":
            self._execute_loot(self.current_action)
        elif action_type == "use_item":
            self._execute_use_item(self.current_action)
        elif action_type == "jump":
            self._execute_jump(self.current_action)
        elif action_type == "wait":
            # Just wait for the duration
            pass
        elif action_type == "stop_movement":
            self._stop_all_movement()
        elif action_type == "key_press":
            self._execute_key_press(self.current_action)
        elif action_type == "mouse_click":
            self._execute_mouse_click(self.current_action)
        else:
            self.logger.warning(f"Unknown action type: {action_type}")
            # Skip this action
            self._finish_current_action()
        
        # Update last action time
        self.last_action_time = current_time
    
    def _finish_current_action(self) -> None:
        """
        Finish the current action and clean up
        """
        if not self.current_action:
            return
        
        action_type = self.current_action.get("type", "")
        
        # Perform any cleanup required by the action type
        if action_type == "move":
            # Stop movement when the action is done
            self._stop_all_movement()
        
        # Clear current action
        self.current_action = None
    
    def _execute_move(self, action: Dict) -> None:
        """
        Execute a movement action
        
        Args:
            action: Movement action dictionary
        """
        # Check if this is a position-based or direction-based movement
        if "position" in action:
            # Position-based movement is more complex
            # In a real implementation, this would navigate the player to a specific position
            # For this simplified implementation, just move in a direction for a time
            
            # Calculate movement direction (this would use real positioning in a full implementation)
            direction = action.get("direction", "forward")
            
            # Start movement in the specified direction
            if direction == "forward":
                self._start_moving_forward()
            elif direction == "backward":
                self._start_moving_backward()
            elif direction == "left":
                self._start_strafing_left()
            elif direction == "right":
                self._start_strafing_right()
            
        elif "direction" in action:
            # Direction-based movement
            direction = action.get("direction")
            
            # Start movement in the specified direction
            if direction == "forward":
                self._start_moving_forward()
            elif direction == "backward":
                self._start_moving_backward()
            elif direction == "left":
                self._start_strafing_left()
            elif direction == "right":
                self._start_strafing_right()
        
        # Movement will stop after the action duration
    
    def _execute_turn(self, action: Dict) -> None:
        """
        Execute a turning action
        
        Args:
            action: Turn action dictionary
        """
        angle = action.get("angle", 0)
        
        if angle < 0:
            # Turn left
            self._start_turning_left()
        else:
            # Turn right
            self._start_turning_right()
        
        # Turning will stop after the action duration
    
    def _execute_cast(self, action: Dict) -> None:
        """
        Execute a spell cast action
        
        Args:
            action: Cast action dictionary
        """
        spell = action.get("spell", "")
        target = action.get("target", "")
        
        # If target specified and not current target, target first
        if target and target != "target" and target != "self":
            self._execute_target({"type": "target", "target": target})
        
        # Find the ability key binding
        ability_key = self._get_ability_key(spell)
        
        if ability_key:
            # Press the ability key
            self._press_key(ability_key)
            self.logger.info(f"Cast {spell} using key {ability_key}")
        else:
            self.logger.warning(f"No keybinding found for spell: {spell}")
    
    def _execute_target(self, action: Dict) -> None:
        """
        Execute a targeting action
        
        Args:
            action: Target action dictionary
        """
        target = action.get("target", "")
        
        if target == "nearest_enemy":
            # Use tab targeting
            self._press_key(self.keybindings.get("target_nearest_enemy", "tab"))
        elif target == "nearest_friend":
            # Use friendly targeting
            self._press_key(self.keybindings.get("target_nearest_friend", "f"))
        else:
            # In a real implementation, this would click on the specific entity
            # For this simplified implementation, just use tab targeting as fallback
            self._press_key(self.keybindings.get("target_nearest_enemy", "tab"))
            
            # Simulate clicking on specific target
            # This would require actual screen coordinates in a real implementation
            self._click_mouse(random.randint(500, 1000), random.randint(300, 600), button="left")
    
    def _execute_interact(self, action: Dict) -> None:
        """
        Execute an interaction action
        
        Args:
            action: Interact action dictionary
        """
        target = action.get("target", "")
        
        # In a real implementation, this would locate the target on screen and click it
        # For this simplified implementation, just right-click in a plausible location
        self._click_mouse(random.randint(500, 1000), random.randint(300, 600), button="right")
    
    def _execute_loot(self, action: Dict) -> None:
        """
        Execute a looting action
        
        Args:
            action: Loot action dictionary
        """
        target = action.get("target", "")
        
        # In a real implementation, this would locate the corpse on screen and right-click it
        # For this simplified implementation, just right-click in a plausible location
        self._click_mouse(random.randint(500, 1000), random.randint(300, 600), button="right")
    
    def _execute_use_item(self, action: Dict) -> None:
        """
        Execute an item use action
        
        Args:
            action: Use item action dictionary
        """
        item = action.get("item", "")
        
        # In a real implementation, this would find the item in bags and use it
        # For this simplified implementation, just simulate opening bags and clicking
        
        # Open bags
        self._press_key(self.keybindings.get("open_bags", "b"))
        time.sleep(0.2)
        
        # Click in bag area
        self._click_mouse(random.randint(800, 1000), random.randint(400, 600), button="right")
        
        # Close bags
        self._press_key(self.keybindings.get("open_bags", "b"))
    
    def _execute_jump(self, action: Dict) -> None:
        """
        Execute a jump action
        
        Args:
            action: Jump action dictionary
        """
        self._press_key(self.keybindings.get("jump", "space"))
    
    def _execute_key_press(self, action: Dict) -> None:
        """
        Execute a key press action
        
        Args:
            action: Key press action dictionary
        """
        key = action.get("key", "")
        if key:
            self._press_key(key)
    
    def _execute_mouse_click(self, action: Dict) -> None:
        """
        Execute a mouse click action
        
        Args:
            action: Mouse click action dictionary
        """
        x = action.get("x", 0)
        y = action.get("y", 0)
        button = action.get("button", "left")
        
        self._click_mouse(x, y, button)
    
    def _start_moving_forward(self) -> None:
        """
        Start moving forward
        """
        if not self.is_moving_forward:
            self.is_moving_forward = True
            self._press_key_down(self.keybindings.get("move_forward", "w"))
    
    def _stop_moving_forward(self) -> None:
        """
        Stop moving forward
        """
        if self.is_moving_forward:
            self.is_moving_forward = False
            self._press_key_up(self.keybindings.get("move_forward", "w"))
    
    def _start_moving_backward(self) -> None:
        """
        Start moving backward
        """
        if not self.is_moving_backward:
            self.is_moving_backward = True
            self._press_key_down(self.keybindings.get("move_backward", "s"))
    
    def _stop_moving_backward(self) -> None:
        """
        Stop moving backward
        """
        if self.is_moving_backward:
            self.is_moving_backward = False
            self._press_key_up(self.keybindings.get("move_backward", "s"))
    
    def _start_strafing_left(self) -> None:
        """
        Start strafing left
        """
        if not self.is_strafing_left:
            self.is_strafing_left = True
            self._press_key_down(self.keybindings.get("strafe_left", "q"))
    
    def _stop_strafing_left(self) -> None:
        """
        Stop strafing left
        """
        if self.is_strafing_left:
            self.is_strafing_left = False
            self._press_key_up(self.keybindings.get("strafe_left", "q"))
    
    def _start_strafing_right(self) -> None:
        """
        Start strafing right
        """
        if not self.is_strafing_right:
            self.is_strafing_right = True
            self._press_key_down(self.keybindings.get("strafe_right", "e"))
    
    def _stop_strafing_right(self) -> None:
        """
        Stop strafing right
        """
        if self.is_strafing_right:
            self.is_strafing_right = False
            self._press_key_up(self.keybindings.get("strafe_right", "e"))
    
    def _start_turning_left(self) -> None:
        """
        Start turning left
        """
        if not self.is_turning_left:
            self.is_turning_left = True
            self._press_key_down(self.keybindings.get("turn_left", "a"))
    
    def _stop_turning_left(self) -> None:
        """
        Stop turning left
        """
        if self.is_turning_left:
            self.is_turning_left = False
            self._press_key_up(self.keybindings.get("turn_left", "a"))
    
    def _start_turning_right(self) -> None:
        """
        Start turning right
        """
        if not self.is_turning_right:
            self.is_turning_right = True
            self._press_key_down(self.keybindings.get("turn_right", "d"))
    
    def _stop_turning_right(self) -> None:
        """
        Stop turning right
        """
        if self.is_turning_right:
            self.is_turning_right = False
            self._press_key_up(self.keybindings.get("turn_right", "d"))
    
    def _stop_all_movement(self) -> None:
        """
        Stop all movement
        """
        self._stop_moving_forward()
        self._stop_moving_backward()
        self._stop_strafing_left()
        self._stop_strafing_right()
        self._stop_turning_left()
        self._stop_turning_right()
    
    def _press_key(self, key: str) -> None:
        """
        Press and release a key
        
        Args:
            key: Key to press
        """
        try:
            # Convert string key representation to pynput key
            pynput_key = self._convert_to_pynput_key(key)
            
            if pynput_key:
                # Press and release
                self.keyboard.press(pynput_key)
                time.sleep(self.key_press_duration)
                self.keyboard.release(pynput_key)
                time.sleep(self.input_delay)
                
                self.logger.debug(f"Pressed key: {key}")
        except Exception as e:
            self.logger.error(f"Error pressing key {key}: {e}")
    
    def _press_key_down(self, key: str) -> None:
        """
        Press a key down (without releasing)
        
        Args:
            key: Key to press down
        """
        try:
            # Convert string key representation to pynput key
            pynput_key = self._convert_to_pynput_key(key)
            
            if pynput_key:
                # Press without release
                self.keyboard.press(pynput_key)
                
                self.logger.debug(f"Pressed key down: {key}")
        except Exception as e:
            self.logger.error(f"Error pressing key down {key}: {e}")
    
    def _press_key_up(self, key: str) -> None:
        """
        Release a key
        
        Args:
            key: Key to release
        """
        try:
            # Convert string key representation to pynput key
            pynput_key = self._convert_to_pynput_key(key)
            
            if pynput_key:
                # Release the key
                self.keyboard.release(pynput_key)
                
                self.logger.debug(f"Released key: {key}")
        except Exception as e:
            self.logger.error(f"Error releasing key {key}: {e}")
            
    def _click_mouse(self, x: int, y: int, button: str = "left") -> None:
        """
        Click mouse at specified coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click (left, right, middle)
        """
        try:
            # Convert coordinate to game window
            x_adjusted = self.window_position[0] + x
            y_adjusted = self.window_position[1] + y
            
            # Move mouse to position
            pyautogui.moveTo(x_adjusted, y_adjusted, duration=self.mouse_move_speed)
            
            # Click the appropriate button
            if button == "left":
                pyautogui.click()
            elif button == "right":
                pyautogui.rightClick()
            elif button == "middle":
                pyautogui.middleClick()
                
            self.logger.debug(f"Clicked {button} mouse button at ({x}, {y})")
            
        except Exception as e:
            self.logger.error(f"Error clicking mouse at ({x}, {y}): {e}")
    
    def move_mouse(self, x: int, y: int, relative: bool = False) -> None:
        """
        Move mouse to position or by a relative amount
        
        Args:
            x: X coordinate or relative movement
            y: Y coordinate or relative movement
            relative: If True, moves mouse relative to current position
        """
        try:
            if relative:
                # Move mouse by relative amount
                current_pos = pyautogui.position()
                new_x = current_pos[0] + x
                new_y = current_pos[1] + y
                pyautogui.moveTo(new_x, new_y, duration=self.mouse_move_speed)
            else:
                # Convert coordinate to game window
                x_adjusted = self.window_position[0] + x
                y_adjusted = self.window_position[1] + y
                
                # Move mouse to absolute position
                pyautogui.moveTo(x_adjusted, y_adjusted, duration=self.mouse_move_speed)
                
            self.logger.debug(f"Moved mouse to {'relative' if relative else 'absolute'} position: ({x}, {y})")
            
        except Exception as e:
            self.logger.error(f"Error moving mouse to ({x}, {y}): {e}")
            
    def _convert_to_pynput_key(self, key: str) -> Any:
        """
        Convert string key representation to pynput key
        
        Args:
            key: String key representation
            
        Returns:
            pynput key object
        """
        # Special keys mapping
        special_keys = {
            "ctrl": keyboard.Key.ctrl,
            "shift": keyboard.Key.shift,
            "alt": keyboard.Key.alt,
            "enter": keyboard.Key.enter,
            "space": keyboard.Key.space,
            "tab": keyboard.Key.tab,
            "esc": keyboard.Key.esc,
            "up": keyboard.Key.up,
            "down": keyboard.Key.down,
            "left": keyboard.Key.left,
            "right": keyboard.Key.right,
            "backspace": keyboard.Key.backspace,
            "delete": keyboard.Key.delete,
            "home": keyboard.Key.home,
            "end": keyboard.Key.end,
            "page_up": keyboard.Key.page_up,
            "page_down": keyboard.Key.page_down,
            "f1": keyboard.Key.f1,
            "f2": keyboard.Key.f2,
            "f3": keyboard.Key.f3,
            "f4": keyboard.Key.f4,
            "f5": keyboard.Key.f5,
            "f6": keyboard.Key.f6,
            "f7": keyboard.Key.f7,
            "f8": keyboard.Key.f8,
            "f9": keyboard.Key.f9,
            "f10": keyboard.Key.f10,
            "f11": keyboard.Key.f11,
            "f12": keyboard.Key.f12
        }
        
        # Check if it's a special key
        if key.lower() in special_keys:
            return special_keys[key.lower()]
            
        # For regular keys, just use the character
        if len(key) == 1:
            return key
            
        # If key is not recognized, log warning and return None
        self.logger.warning(f"Unrecognized key: {key}")
        return None
            
    def _get_ability_key(self, ability_name: str) -> str:
        """
        Get keybinding for an ability
        
        Args:
            ability_name: Name of the ability
            
        Returns:
            Key binding for the ability
        """
        # In a real implementation, this would look up the ability in a mapping
        # For this simplified implementation, just map to action bar slots
        
        # Check if ability is directly in keybindings
        if ability_name in self.keybindings:
            return self.keybindings[ability_name]
            
        # Default mapping of some common abilities to action bar slots
        ability_defaults = {
            # Warrior
            "Charge": "1",
            "Heroic Strike": "2",
            "Thunder Clap": "3",
            "Rend": "4",
            "Mortal Strike": "5",
            "Execute": "6",
            "Whirlwind": "7",
            "Hamstring": "8",
            "Overpower": "9",
            
            # Mage
            "Fireball": "1",
            "Frostbolt": "2",
            "Arcane Missiles": "3",
            "Frost Nova": "4",
            "Blink": "5",
            "Polymorph": "6",
            "Ice Block": "7",
            "Counterspell": "8",
            "Evocation": "9",
            
            # Priest
            "Smite": "1",
            "Shadow Word: Pain": "2",
            "Power Word: Shield": "3",
            "Renew": "4",
            "Mind Blast": "5",
            "Flash Heal": "6",
            "Greater Heal": "7",
            "Psychic Scream": "8",
            "Fade": "9",
            
            # Generic
            "Auto Attack": "1",
            "Attack": "1"
        }
        
        # Return the default keybinding for the ability, or the first action bar slot as fallback
        return ability_defaults.get(ability_name, "1")