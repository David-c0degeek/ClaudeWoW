"""
UI Detector Module

This module is responsible for detecting and analyzing UI elements in the game.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import json
import os

class UIDetector:
    """
    Detects and analyzes UI elements in World of Warcraft screenshots
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the UIDetector
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.perception.ui_detector")
        self.config = config
        self.resolution = (config["resolution"]["width"], config["resolution"]["height"])
        self.ui_scale = config.get("ui_scale", 1.0)
        
        # Load UI templates
        self.templates = self._load_templates()
        
        # Load UI element locations from config or use defaults based on resolution
        self.ui_elements_config = self._load_ui_elements_config()
        
        self.logger.info("UIDetector initialized")
    
    def _load_templates(self) -> Dict[str, np.ndarray]:
        """
        Load template images for UI detection
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of template images
        """
        templates = {}
        templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "templates"
        )
        
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
            self.logger.warning(f"Templates directory {templates_dir} created, but no templates found")
            return templates
        
        for filename in os.listdir(templates_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    template_path = os.path.join(templates_dir, filename)
                    template_name = os.path.splitext(filename)[0]
                    template_img = cv2.imread(template_path)
                    if template_img is not None:
                        templates[template_name] = template_img
                        self.logger.debug(f"Loaded template: {template_name}")
                    else:
                        self.logger.warning(f"Failed to load template: {template_path}")
                except Exception as e:
                    self.logger.error(f"Error loading template {filename}: {e}")
        
        return templates
    
    def detect_ui_elements(self, screenshot: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Detect UI elements in the screenshot
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            Dict: Detected UI elements with their coordinates (x, y, width, height)
        """
        results = {}
        
        # First use template matching for dynamic elements
        results.update(self._detect_dynamic_elements(screenshot))
        
        # Then use predefined locations for static elements
        for element_name, coords in self.ui_elements_config.items():
            if element_name not in results:  # Don't override template matches
                x = coords["x"]
                y = coords["y"]
                width = coords["width"]
                height = coords["height"]
                
                # Apply UI scale factor if needed
                if self.ui_scale != 1.0:
                    x = int(x * self.ui_scale)
                    y = int(y * self.ui_scale)
                    width = int(width * self.ui_scale)
                    height = int(height * self.ui_scale)
                
                results[element_name] = (x, y, width, height)
        
        return results
    
    def _detect_dynamic_elements(self, screenshot: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Detect dynamic UI elements using template matching
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            Dict: Detected dynamic UI elements
        """
        results = {}
        
        # Convert screenshot to grayscale for template matching
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        
        # Perform template matching for each template
        for template_name, template in self.templates.items():
            try:
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                match_result = cv2.matchTemplate(gray_screenshot, gray_template, cv2.TM_CCOEFF_NORMED)
                
                # Get the best match location
                _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
                
                # If match is good enough, add to results
                if max_val > 0.8:  # Threshold for considering a good match
                    x, y = max_loc
                    h, w = template.shape[:2]
                    results[template_name] = (x, y, w, h)
                    self.logger.debug(f"Detected {template_name} at {max_loc} with confidence {max_val:.2f}")
            except Exception as e:
                self.logger.error(f"Error matching template {template_name}: {e}")
        
        return results
    
    def get_bar_percentage(self, screenshot: np.ndarray, bar_rect: Tuple[int, int, int, int]) -> float:
        """
        Calculate the percentage of a bar (health, mana, etc.)
        
        Args:
            screenshot: The game screenshot
            bar_rect: Rectangle coordinates of the bar (x, y, width, height)
        
        Returns:
            float: Percentage of the bar (0.0 to 100.0)
        """
        try:
            x, y, w, h = bar_rect
            
            # Extract the bar region
            bar_region = screenshot[y:y+h, x:x+w]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(bar_region, cv2.COLOR_RGB2HSV)
            
            # Detect filled portion (depends on bar color)
            # Assuming health bar is red/green and mana bar is blue
            # This is simplified and would need to be improved for actual implementation
            
            # Try to detect health bar (red/green)
            lower_health = np.array([0, 100, 100])
            upper_health = np.array([70, 255, 255])
            health_mask = cv2.inRange(hsv, lower_health, upper_health)
            
            # Try to detect mana bar (blue)
            lower_mana = np.array([90, 100, 100])
            upper_mana = np.array([150, 255, 255])
            mana_mask = cv2.inRange(hsv, lower_mana, upper_mana)
            
            # Combine masks
            mask = cv2.bitwise_or(health_mask, mana_mask)
            
            # Count non-zero pixels (filled portion)
            filled_pixels = cv2.countNonZero(mask)
            total_pixels = w * h
            
            # Calculate percentage
            if total_pixels > 0:
                percentage = (filled_pixels / total_pixels) * 100.0
                return percentage
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating bar percentage: {e}")
            return 0.0
    
    def detect_buffs(self, screenshot: np.ndarray, area_rect: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Detect buffs in the specified area
        
        Args:
            screenshot: The game screenshot
            area_rect: Rectangle coordinates of the buffs area
        
        Returns:
            List[Dict]: Detected buffs with positions and durations
        """
        # This is a placeholder implementation
        # A real implementation would use more sophisticated image processing
        # to detect individual buff icons and their timers
        buffs = []
        
        try:
            x, y, w, h = area_rect
            area = screenshot[y:y+h, x:x+w]
            
            # Simple approach: look for square-ish colored regions that might be buff icons
            gray = cv2.cvtColor(area, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                
                # Check if it's square-ish (buff icons are usually square)
                aspect_ratio = float(w_c) / h_c if h_c > 0 else 0
                
                if 0.8 <= aspect_ratio <= 1.2 and w_c >= 20 and h_c >= 20:
                    # This might be a buff icon
                    buff = {
                        "id": f"buff_{i}",
                        "position": (x + x_c, y + y_c),
                        "size": (w_c, h_c),
                        # In a real implementation, we would extract the buff duration text
                        "duration": None  
                    }
                    buffs.append(buff)
            
        except Exception as e:
            self.logger.error(f"Error detecting buffs: {e}")
        
        return buffs
    
    def detect_debuffs(self, screenshot: np.ndarray, area_rect: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Detect debuffs in the specified area
        
        Args:
            screenshot: The game screenshot
            area_rect: Rectangle coordinates of the debuffs area
        
        Returns:
            List[Dict]: Detected debuffs with positions and durations
        """
        # Similar to detect_buffs, this is a placeholder
        # The real implementation would be more sophisticated
        return self.detect_buffs(screenshot, area_rect)  # Reuse the same logic for now
    
    def detect_cooldowns(self, screenshot: np.ndarray, action_bars_rect: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Detect ability cooldowns on action bars
        
        Args:
            screenshot: The game screenshot
            action_bars_rect: Rectangle coordinates of the action bars
        
        Returns:
            Dict[str, float]: Dictionary of ability positions and their cooldown percentages
        """
        cooldowns = {}
        
        try:
            x, y, w, h = action_bars_rect
            bars_area = screenshot[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(bars_area, cv2.COLOR_RGB2GRAY)
            
            # Look for the cooldown swipe animation (darker overlay on abilities)
            # This is a simplified approach
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours of potential ability slots
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                
                # Check if it looks like an ability slot (square)
                aspect_ratio = float(w_c) / h_c if h_c > 0 else 0
                
                if 0.8 <= aspect_ratio <= 1.2 and w_c >= 30 and h_c >= 30:
                    # Extract the ability slot
                    slot = bars_area[y_c:y_c+h_c, x_c:x_c+w_c]
                    
                    # Check for cooldown swipe (darker region)
                    # A real implementation would use more sophisticated approaches
                    slot_gray = cv2.cvtColor(slot, cv2.COLOR_RGB2GRAY) if len(slot.shape) == 3 else slot
                    avg_brightness = np.mean(slot_gray)
                    
                    # If average brightness is low, might be in cooldown
                    if avg_brightness < 100:
                        # Estimate cooldown percentage based on brightness
                        # This is very simplified
                        cooldown_pct = 1.0 - (avg_brightness / 255.0)
                        ability_id = f"ability_{i}"
                        cooldowns[ability_id] = cooldown_pct * 100.0
            
        except Exception as e:
            self.logger.error(f"Error detecting cooldowns: {e}")
        
        return cooldowns
    
    def detect_cast_bar(self, screenshot: np.ndarray, cast_bar_rect: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Detect and analyze cast bar
        
        Args:
            screenshot: The game screenshot
            cast_bar_rect: Rectangle coordinates of the cast bar
        
        Returns:
            Optional[Dict]: Cast information if casting, None otherwise
        """
        try:
            x, y, w, h = cast_bar_rect
            
            # Extract the cast bar region
            cast_region = screenshot[y:y+h, x:x+w]
            
            # Check if cast bar is present (not empty)
            gray = cv2.cvtColor(cast_region, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)
            
            # If average brightness is very low, probably no cast bar is present
            if avg_brightness < 30:
                return None
            
            # Detect the fill level of the cast bar
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = w * h
            
            if total_pixels > 0:
                progress = white_pixels / total_pixels
            else:
                progress = 0
            
            # Extract cast name (would require OCR in a real implementation)
            # This is a placeholder
            cast_name = "Unknown Cast"
            
            return {
                "name": cast_name,
                "progress": progress * 100.0,
                "position": (x, y),
                "size": (w, h)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting cast bar: {e}")
            return None
    
    def is_in_combat(self, screenshot: np.ndarray, ui_elements: Dict) -> bool:
        """
        Determine if the player is in combat
        
        Args:
            screenshot: The game screenshot
            ui_elements: Detected UI elements
        
        Returns:
            bool: True if in combat, False otherwise
        """
        # In WoW, the player portrait has a red glow when in combat
        # This is a simplified implementation
        try:
            if "player_frame" in ui_elements:
                x, y, w, h = ui_elements["player_frame"]
                frame_region = screenshot[y:y+h, x:x+w]
                
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(frame_region, cv2.COLOR_RGB2HSV)
                
                # Define range for red color (combat glow)
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 100, 100])
                upper_red2 = np.array([180, 255, 255])
                
                # Create masks for red detection
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)
                
                # Count red pixels
                red_pixels = cv2.countNonZero(mask)
                
                # If significant number of red pixels, probably in combat
                return red_pixels > (w * h * 0.1)
        except Exception as e:
            self.logger.error(f"Error detecting combat state: {e}")
        
        return False
    
    def is_mounted(self, screenshot: np.ndarray, ui_elements: Dict) -> bool:
        """
        Determine if the player is mounted
        
        Args:
            screenshot: The game screenshot
            ui_elements: Detected UI elements
        
        Returns:
            bool: True if mounted, False otherwise
        """
        # In WoW, a mount buff appears in the buff bar when mounted
        # This is a simplified implementation that would look for mount buffs
        try:
            if "player_buffs_area" in ui_elements:
                buffs = self.detect_buffs(screenshot, ui_elements["player_buffs_area"])
                
                # In a real implementation, we would check for specific mount buff icons
                # This is a placeholder that would need to be improved
                return len(buffs) > 0 and any("mount" in buff.get("id", "").lower() for buff in buffs)
        except Exception as e:
            self.logger.error(f"Error detecting mount state: {e}")
        
        return False
    
    def is_resting(self, screenshot: np.ndarray, ui_elements: Dict) -> bool:
        """
        Determine if the player is resting (in an inn or city)
        
        Args:
            screenshot: The game screenshot
            ui_elements: Detected UI elements
        
        Returns:
            bool: True if resting, False otherwise
        """
        # In WoW, a "resting" icon appears near the character portrait
        # This is a simplified implementation
        try:
            if "player_frame" in ui_elements:
                # A real implementation would use template matching with a resting icon template
                # This is just a placeholder
                return False
        except Exception as e:
            self.logger.error(f"Error detecting resting state: {e}")
        
        return False
    
    def is_in_group(self, screenshot: np.ndarray, ui_elements: Dict) -> bool:
        """
        Determine if the player is in a group
        
        Args:
            screenshot: The game screenshot
            ui_elements: Detected UI elements
        
        Returns:
            bool: True if in a group, False otherwise
        """
        # Look for party frames
        # This is a placeholder implementation
        return False
    
    def is_in_raid(self, screenshot: np.ndarray, ui_elements: Dict) -> bool:
        """
        Determine if the player is in a raid
        
        Args:
            screenshot: The game screenshot
            ui_elements: Detected UI elements
        
        Returns:
            bool: True if in a raid, False otherwise
        """
        # Look for raid frames
        # This is a placeholder implementation
        return False
    
    def is_in_dungeon(self, screenshot: np.ndarray, ui_elements: Dict) -> bool:
        """
        Determine if the player is in a dungeon/instance
        
        Args:
            screenshot: The game screenshot
            ui_elements: Detected UI elements
        
        Returns:
            bool: True if in a dungeon, False otherwise
        """
        # Could check zone text or instance portal minimap icon
        # This is a placeholder implementation
        return False
    
    def _load_ui_elements_config(self) -> Dict:
        """
        Load UI element configurations from file or use defaults
        
        Returns:
            Dict: UI element configurations
        """
        ui_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "ui_elements.json"
        )
        
        # Default configurations based on resolution
        default_config = self._create_default_ui_config()
        
        if not os.path.exists(ui_config_path):
            # Save default config
            os.makedirs(os.path.dirname(ui_config_path), exist_ok=True)
            with open(ui_config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            self.logger.info(f"Created default UI config at {ui_config_path}")
            return default_config
        
        try:
            with open(ui_config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded UI config from {ui_config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading UI config: {e}, using defaults")
            return default_config
    
    def _create_default_ui_config(self) -> Dict:
        """
        Create default UI element configurations based on screen resolution
        
        Returns:
            Dict: Default UI element configurations
        """
        width, height = self.resolution
        
        # Scale factors for different resolutions
        scale_x = width / 1920
        scale_y = height / 1080
        
        # Default locations for 1920x1080 scaled to current resolution
        return {
            "player_frame": {
                "x": int(265 * scale_x),
                "y": int(43 * scale_y),
                "width": int(232 * scale_x),
                "height": int(100 * scale_y)
            },
            "player_health_bar": {
                "x": int(106 * scale_x),
                "y": int(65 * scale_y),
                "width": int(180 * scale_x),
                "height": int(20 * scale_y)
            },
            "target_name_text": {
                "x": int(1432 * scale_x),
                "y": int(43 * scale_y),
                "width": int(180 * scale_x),
                "height": int(20 * scale_y)
            },
            "target_type_text": {
                "x": int(1432 * scale_x),
                "y": int(43 * scale_y),
                "width": int(180 * scale_x),
                "height": int(20 * scale_y)
            },
            "target_buffs_area": {
                "x": int(1432 * scale_x),
                "y": int(43 * scale_y),
                "width": int(500 * scale_x),
                "height": int(35 * scale_y)
            },
            "target_debuffs_area": {
                "x": int(1432 * scale_x),
                "y": int(79 * scale_y),
                "width": int(500 * scale_x),
                "height": int(35 * scale_y)
            },
            "target_cast_bar": {
                "x": int(660 * scale_x),
                "y": int(450 * scale_y),
                "width": int(600 * scale_x),
                "height": int(32 * scale_y)
            },
            "action_bars": {
                "x": int(632 * scale_x),
                "y": int(900 * scale_y),
                "width": int(656 * scale_x),
                "height": int(160 * scale_y)
            },
            "minimap": {
                "x": int(1720 * scale_x),
                "y": int(43 * scale_y),
                "width": int(180 * scale_x),
                "height": int(180 * scale_y)
            },
            "zone_text": {
                "x": int(1720 * scale_x),
                "y": int(25 * scale_y),
                "width": int(180 * scale_x),
                "height": int(18 * scale_y)
            },
            "quest_tracker": {
                "x": int(1620 * scale_x),
                "y": int(250 * scale_y),
                "width": int(280 * scale_x),
                "height": int(400 * scale_y)
            },
            "chat_window": {
                "x": int(20 * scale_x),
                "y": int(860 * scale_y),
                "width": int(480 * scale_x),
                "height": int(200 * scale_y)
            }
        }