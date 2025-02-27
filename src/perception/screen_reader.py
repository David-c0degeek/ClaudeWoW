"""
Screen Reader Module

This module is responsible for capturing and interpreting the game screen.
It uses computer vision techniques to extract game state information from pixels.
"""

import logging
import numpy as np
import cv2
import pytesseract
import pyautogui
from typing import Dict, List, Tuple, Any
import time
import os

from src.utils.image_processing import preprocess_image, apply_ocr_preprocessing
from src.perception.ui_detector import UIDetector
from src.perception.entity_detector import EntityDetector
from src.perception.text_extractor import TextExtractor
from src.perception.minimap_analyzer import MinimapAnalyzer

class GameState:
    """Class representing the current state of the game"""
    
    def __init__(self):
        self.player_health = 100.0
        self.player_mana = 100.0
        self.player_level = 1
        self.player_position = (0, 0)
        self.player_class = ""
        self.player_race = ""
        self.player_talents = {}
        self.player_buffs = []
        self.player_debuffs = []
        self.player_cooldowns = {}
        self.player_inventory = {}
        self.player_gold = 0
        
        self.target = None
        self.target_health = 0.0
        self.target_type = ""  # NPC, enemy, friendly, etc.
        self.target_buffs = []
        self.target_debuffs = []
        self.target_cast = None
        
        self.nearby_entities = []
        self.nearby_npcs = []
        self.nearby_players = []
        self.nearby_objects = []
        
        self.current_zone = ""
        self.current_subzone = ""
        self.minimap_data = {}
        
        self.current_quest = {}
        self.active_quests = []
        self.completed_quests = []
        
        self.chat_log = []
        self.combat_log = []
        
        self.is_in_combat = False
        self.is_mounted = False
        self.is_resting = False
        self.is_in_group = False
        self.is_in_raid = False
        self.is_in_dungeon = False

        # Social state
        self.chat_log = []          # Recent chat messages
        self.group_members = []     # Current group/raid members
        self.is_group_leader = False  # Whether the player is group leader
        self.nearby_players = []    # Players in vicinity
        self.current_instance = ""  # Current dungeon/raid name
        self.is_in_instance = False  # Whether in a dungeon/raid

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return self.__dict__


class ScreenReader:
    """
    Captures and analyzes the game screen to extract game state information
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ScreenReader
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.perception.screen_reader")
        self.config = config
        self.resolution = (config["resolution"]["width"], config["resolution"]["height"])
        self.ui_scale = config.get("ui_scale", 1.0)
        self.screenshot_interval = config.get("screenshot_interval", 0.1)
        
        # Set pytesseract path if provided in config
        if "tesseract_path" in config:
            pytesseract.pytesseract.tesseract_cmd = config["tesseract_path"]
        
        # Initialize detectors
        self.ui_detector = UIDetector(config)
        self.entity_detector = EntityDetector(config)
        self.text_extractor = TextExtractor(config)
        self.minimap_analyzer = MinimapAnalyzer(config)
        
        # Create data directory for debug screenshots if needed
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "data", "debug_screenshots")
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        self.last_capture_time = 0
        self.current_screenshot = None
        self.logger.info("ScreenReader initialized")
    
    def capture_screenshot(self) -> np.ndarray:
        """
        Capture the current game screen
        
        Returns:
            numpy.ndarray: The screenshot as a numpy array
        """
        current_time = time.time()
        if current_time - self.last_capture_time < self.screenshot_interval:
            return self.current_screenshot
        
        screenshot = pyautogui.screenshot()
        self.current_screenshot = np.array(screenshot)
        self.last_capture_time = current_time
        
        if self.config.get("save_debug_screenshots", False):
            timestamp = int(time.time())
            debug_path = os.path.join(self.debug_dir, f"screenshot_{timestamp}.png")
            cv2.imwrite(debug_path, cv2.cvtColor(self.current_screenshot, cv2.COLOR_RGB2BGR))
        
        return self.current_screenshot
    
    def capture_game_state(self) -> GameState:
        """
        Capture and analyze the current game state
        
        Returns:
            GameState: Object containing all extracted game state information
        """
        self.logger.debug("Capturing game state")
        state = GameState()
        
        # Capture screenshot
        screenshot = self.capture_screenshot()
        if screenshot is None:
            self.logger.warning("Failed to capture screenshot")
            return state
        
        try:
            # Extract UI information
            ui_elements = self.ui_detector.detect_ui_elements(screenshot)
            
            # Update player info
            self._extract_player_info(screenshot, ui_elements, state)
            
            # Extract target info
            self._extract_target_info(screenshot, ui_elements, state)
            
            # Extract environment info
            self._extract_environment_info(screenshot, ui_elements, state)
            
            # Extract quest info
            self._extract_quest_info(screenshot, ui_elements, state)
            
            # Extract chat and combat log
            self._extract_chat_and_combat_log(screenshot, ui_elements, state)
            
            # Detect nearby entities
            self._detect_nearby_entities(screenshot, state)
            
            # Analyze minimap
            minimap_data = self.minimap_analyzer.analyze_minimap(
                screenshot, ui_elements.get("minimap_rect"))
            state.minimap_data = minimap_data
            
        except Exception as e:
            self.logger.error(f"Error capturing game state: {e}")
            self.logger.exception(e)
        
        return state
    
    def _extract_player_info(self, screenshot: np.ndarray, 
                            ui_elements: Dict, state: GameState) -> None:
        """Extract player information from the UI"""
        # Health and mana from UI bars
        if "player_health_bar" in ui_elements:
            state.player_health = self.ui_detector.get_bar_percentage(
                screenshot, ui_elements["player_health_bar"])
        
        if "player_mana_bar" in ui_elements:
            state.player_mana = self.ui_detector.get_bar_percentage(
                screenshot, ui_elements["player_mana_bar"])
        
        # Extract player level from UI
        if "player_level_text" in ui_elements:
            level_text = self.text_extractor.extract_text(
                screenshot, ui_elements["player_level_text"])
            try:
                state.player_level = int(level_text)
            except ValueError:
                pass
        
        # Extract buffs and debuffs
        if "player_buffs_area" in ui_elements:
            state.player_buffs = self.ui_detector.detect_buffs(
                screenshot, ui_elements["player_buffs_area"])
        
        if "player_debuffs_area" in ui_elements:
            state.player_debuffs = self.ui_detector.detect_debuffs(
                screenshot, ui_elements["player_debuffs_area"])
        
        # Extract cooldowns from action bars
        if "action_bars" in ui_elements:
            state.player_cooldowns = self.ui_detector.detect_cooldowns(
                screenshot, ui_elements["action_bars"])
        
        # Extract combat state
        state.is_in_combat = self.ui_detector.is_in_combat(screenshot, ui_elements)
        
        # Additional states
        state.is_mounted = self.ui_detector.is_mounted(screenshot, ui_elements)
        state.is_resting = self.ui_detector.is_resting(screenshot, ui_elements)
    
    def _extract_target_info(self, screenshot: np.ndarray, 
                           ui_elements: Dict, state: GameState) -> None:
        """Extract information about the current target"""
        if "target_frame" not in ui_elements:
            state.target = None
            return
        
        # Extract target health
        if "target_health_bar" in ui_elements:
            state.target_health = self.ui_detector.get_bar_percentage(
                screenshot, ui_elements["target_health_bar"])
        
        # Extract target name and type
        if "target_name_text" in ui_elements:
            target_name = self.text_extractor.extract_text(
                screenshot, ui_elements["target_name_text"])
            state.target = target_name
        
        if "target_type_text" in ui_elements:
            target_type = self.text_extractor.extract_text(
                screenshot, ui_elements["target_type_text"])
            state.target_type = target_type
        
        # Extract target buffs/debuffs
        if "target_buffs_area" in ui_elements:
            state.target_buffs = self.ui_detector.detect_buffs(
                screenshot, ui_elements["target_buffs_area"])
        
        if "target_debuffs_area" in ui_elements:
            state.target_debuffs = self.ui_detector.detect_debuffs(
                screenshot, ui_elements["target_debuffs_area"])
        
        # Extract target cast bar
        if "target_cast_bar" in ui_elements:
            cast_info = self.ui_detector.detect_cast_bar(
                screenshot, ui_elements["target_cast_bar"])
            state.target_cast = cast_info
    
    def _extract_environment_info(self, screenshot: np.ndarray, 
                                ui_elements: Dict, state: GameState) -> None:
        """Extract information about the game environment"""
        # Extract zone information
        if "zone_text" in ui_elements:
            zone_text = self.text_extractor.extract_text(
                screenshot, ui_elements["zone_text"])
            state.current_zone = zone_text
        
        if "subzone_text" in ui_elements:
            subzone_text = self.text_extractor.extract_text(
                screenshot, ui_elements["subzone_text"])
            state.current_subzone = subzone_text
        
        # Check if in group/raid
        state.is_in_group = self.ui_detector.is_in_group(screenshot, ui_elements)
        state.is_in_raid = self.ui_detector.is_in_raid(screenshot, ui_elements)
        
        # Check if in dungeon/instance
        state.is_in_dungeon = self.ui_detector.is_in_dungeon(screenshot, ui_elements)
    
    def _extract_quest_info(self, screenshot: np.ndarray, 
                           ui_elements: Dict, state: GameState) -> None:
        """Extract quest information"""
        # Extract current quest objectives
        if "quest_tracker" in ui_elements:
            quest_text = self.text_extractor.extract_text(
                screenshot, ui_elements["quest_tracker"], multiline=True)
            
            # Simple parsing of quest text
            quest_lines = quest_text.strip().split('\n')
            current_quest = {}
            active_quests = []
            
            for i, line in enumerate(quest_lines):
                if i == 0 and line:  # First line might be the quest title
                    current_quest["title"] = line
                elif ":" in line:  # Objective line
                    obj_parts = line.split(":")
                    if len(obj_parts) == 2:
                        obj_name = obj_parts[0].strip()
                        obj_progress = obj_parts[1].strip()
                        
                        # Try to parse progress (e.g., "5/10")
                        progress_parts = obj_progress.split('/')
                        if len(progress_parts) == 2:
                            try:
                                current = int(progress_parts[0])
                                total = int(progress_parts[1])
                                if "objectives" not in current_quest:
                                    current_quest["objectives"] = []
                                current_quest["objectives"].append({
                                    "name": obj_name,
                                    "current": current,
                                    "total": total
                                })
                            except ValueError:
                                pass
            
            if current_quest:
                state.current_quest = current_quest
                active_quests.append(current_quest)
            
            state.active_quests = active_quests
    
    def _extract_chat_and_combat_log(self, screenshot: np.ndarray, 
                                    ui_elements: Dict, state: GameState) -> None:
        """Extract chat and combat log information"""
        # Extract chat messages
        if "chat_window" in ui_elements:
            chat_text = self.text_extractor.extract_text(
                screenshot, ui_elements["chat_window"], multiline=True)
            
            chat_lines = chat_text.strip().split('\n')
            # Keep only the last 10 lines to avoid state bloat
            state.chat_log = chat_lines[-10:] if chat_lines else []
        
        # Extract combat log if visible
        if "combat_log_window" in ui_elements:
            combat_text = self.text_extractor.extract_text(
                screenshot, ui_elements["combat_log_window"], multiline=True)
            
            combat_lines = combat_text.strip().split('\n')
            # Keep only the last 10 lines
            state.combat_log = combat_lines[-10:] if combat_lines else []

        # Extract chat messages
        if "chat_window" in ui_elements:
            chat_text = self.text_extractor.extract_text(
                screenshot, ui_elements["chat_window"], multiline=True)
        
            chat_lines = chat_text.strip().split('\n')
        
            # Extract only new lines since last check
            new_chat_lines = []
            if hasattr(self, 'last_chat_lines'):
                for line in chat_lines:
                    if line not in self.last_chat_lines:
                        new_chat_lines.append(line)
            else:
                new_chat_lines = chat_lines

            self.last_chat_lines = chat_lines

            # Add new chat lines to state
            state.chat_log = new_chat_lines

            # Extract group information
            if "group_frames" in ui_elements:
                # Process group frames to extract member information
                group_members = self._extract_group_members(screenshot, ui_elements["group_frames"])
                state.group_members = group_members

                # Check if player is leader (has crown icon)
                state.is_group_leader = self._is_player_group_leader(screenshot, ui_elements)
    
    def _detect_nearby_entities(self, screenshot: np.ndarray, state: GameState) -> None:
        """Detect nearby entities using computer vision"""
        entities = self.entity_detector.detect_entities(screenshot)
        
        state.nearby_entities = entities
        
        # Filter entities by type
        state.nearby_npcs = [e for e in entities if e.get("type") == "npc"]
        state.nearby_players = [e for e in entities if e.get("type") == "player"]
        state.nearby_objects = [e for e in entities if e.get("type") == "object"]