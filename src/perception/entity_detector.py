"""
Entity Detector Module

This module is responsible for detecting entities (NPCs, players, objects) in the game world.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
import os
import torch
import time

class EntityDetector:
    """
    Detects and analyzes entities in World of Warcraft screenshots
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the EntityDetector
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.perception.entity_detector")
        self.config = config
        self.resolution = (config["resolution"]["width"], config["resolution"]["height"])
        
        # Load or initialize the entity detection model
        self.model = self._load_model()
        
        # Define entity classes
        self.entity_classes = [
            "player_ally", "player_horde", "npc_friendly", "npc_neutral", 
            "npc_hostile", "mob", "vendor", "quest_giver", "resource_node",
            "object_interactable"
        ]
        
        # Load color templates for nameplates
        self.nameplate_colors = {
            "ally": (0, 255, 0),      # Green
            "horde": (255, 0, 0),     # Red
            "friendly": (0, 255, 0),  # Green
            "neutral": (255, 255, 0), # Yellow
            "hostile": (255, 0, 0)    # Red
        }
        
        self.last_detection_time = 0
        self.detection_interval = config.get("entity_detection_interval", 0.5)
        self.cached_entities = []
        
        self.logger.info("EntityDetector initialized")
    
    def _load_model(self):
        """
        Load the entity detection model
        
        Returns:
            The loaded model or None if not available
        """
        try:
            model_path = self.config.get("model_paths", {}).get("entity_detection")
            
            if model_path and os.path.exists(model_path):
                # In a real implementation, we would load a trained deep learning model
                # This is a placeholder
                self.logger.info(f"Loading entity detection model from {model_path}")
                # model = torch.load(model_path)
                # return model
                return None
            else:
                self.logger.warning("Entity detection model not found, using fallback methods")
                return None
        except Exception as e:
            self.logger.error(f"Error loading entity detection model: {e}")
            return None
    
    def detect_entities(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect entities in the screenshot
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            List[Dict]: Detected entities with their info
        """
        current_time = time.time()
        
        # Use cached detections if within interval
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_entities
        
        self.last_detection_time = current_time
        entities = []
        
        try:
            # If we have a trained model, use it
            if self.model is not None:
                entities = self._detect_with_model(screenshot)
            else:
                # Otherwise, use fallback methods
                entities = self._detect_with_fallback(screenshot)
            
            # Cache the results
            self.cached_entities = entities
            
        except Exception as e:
            self.logger.error(f"Error detecting entities: {e}")
        
        return entities
    
    def _detect_with_model(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect entities using a trained deep learning model
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            List[Dict]: Detected entities
        """
        # This is a placeholder for using a trained model
        # In a real implementation, we would:
        # 1. Preprocess the image
        # 2. Run inference with the model
        # 3. Process and format the results
        
        # Placeholder for model detection
        return []
    
    def _detect_with_fallback(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect entities using fallback computer vision methods
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            List[Dict]: Detected entities
        """
        entities = []
        
        # Detect nameplates (colored text above entities)
        nameplate_entities = self._detect_nameplates(screenshot)
        entities.extend(nameplate_entities)
        
        # Detect player character models
        player_entities = self._detect_player_models(screenshot)
        entities.extend(player_entities)
        
        # Detect NPCs and objects
        npc_entities = self._detect_npc_models(screenshot)
        entities.extend(npc_entities)
        
        return entities
    
    def _detect_nameplates(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect entities by their nameplates
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            List[Dict]: Detected entities
        """
        entities = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
            
            # Create masks for each nameplate color
            masks = {}
            for type_name, color in self.nameplate_colors.items():
                # Convert RGB to HSV
                hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
                
                # Create a color range for the mask
                lower_bound = np.array([hsv_color[0] - 10, 100, 100])
                upper_bound = np.array([hsv_color[0] + 10, 255, 255])
                
                masks[type_name] = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Process each mask to find contours
            for type_name, mask in masks.items():
                # Apply morphological operations to clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Process each contour as a potential nameplate
                for i, contour in enumerate(contours):
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter out very small or large contours
                    if w > 20 and h > 5 and w < 200 and h < 30:
                        # This could be a nameplate
                        # Extend the search area below to look for the entity model
                        entity_x = x
                        entity_y = y + h  # Start below the nameplate
                        entity_w = w
                        entity_h = h * 3  # Approximate height of the entity model
                        
                        entity_type = "player" if type_name in ["ally", "horde"] else "npc"
                        faction = type_name if type_name in ["ally", "horde"] else None
                        reaction = type_name if type_name in ["friendly", "neutral", "hostile"] else None
                        
                        entity = {
                            "id": f"{entity_type}_{type_name}_{i}",
                            "type": entity_type,
                            "faction": faction,
                            "reaction": reaction,
                            "nameplate": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h
                            },
                            "body": {
                                "x": entity_x,
                                "y": entity_y,
                                "width": entity_w,
                                "height": entity_h
                            },
                            "position": (entity_x + entity_w // 2, entity_y + entity_h // 2),
                            "confidence": 0.7  # Confidence score placeholder
                        }
                        
                        entities.append(entity)
        
        except Exception as e:
            self.logger.error(f"Error detecting nameplates: {e}")
        
        return entities
    
    def _detect_player_models(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect player character models using image processing
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            List[Dict]: Detected player entities
        """
        # This would involve more sophisticated computer vision in a real implementation
        # For a proper implementation, we would need:
        # 1. Trained models to recognize player character models
        # 2. Or template matching with player model templates
        # 3. Motion detection to identify moving entities
        
        # This is a placeholder implementation
        return []
    
    def _detect_npc_models(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect NPC models and interactive objects using image processing
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            List[Dict]: Detected NPC entities
        """
        # Similar to player model detection, this would need more sophisticated
        # computer vision techniques in a real implementation
        
        # For quest givers, we could look for the yellow exclamation mark
        quest_givers = self._detect_quest_markers(screenshot)
        
        # For vendors, we could look for specific icons
        vendors = []
        
        # For resource nodes, we could look for specific shapes/colors
        resource_nodes = []
        
        # Combine all detected NPCs and objects
        return quest_givers + vendors + resource_nodes
    
    def _detect_quest_markers(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect quest markers (yellow ! for available quests, yellow ? for completable quests)
        
        Args:
            screenshot: The game screenshot
        
        Returns:
            List[Dict]: Detected quest giver entities
        """
        entities = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
            
            # Define color range for yellow quest markers
            lower_yellow = np.array([25, 150, 150])
            upper_yellow = np.array([35, 255, 255])
            
            # Create mask for yellow color
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour as a potential quest marker
            for i, contour in enumerate(contours):
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter contours that might be quest markers (! or ?)
                if 5 <= w <= 20 and 15 <= h <= 40:
                    # Extract the region to check if it's a ! or ?
                    marker_region = yellow_mask[y:y+h, x:x+w]
                    
                    # Determine if it's a ! or ? based on shape
                    # This is a simplified approach
                    is_exclamation = h > 2*w  # ! is taller and thinner
                    marker_type = "available_quest" if is_exclamation else "completable_quest"
                    
                    # Create entity entry
                    entity = {
                        "id": f"quest_giver_{i}",
                        "type": "npc",
                        "subtype": "quest_giver",
                        "marker": marker_type,
                        "position": (x + w // 2, y + h // 2),
                        "rect": (x, y, w, h),
                        "confidence": 0.8  # Confidence score placeholder
                    }
                    
                    entities.append(entity)
        
        except Exception as e:
            self.logger.error(f"Error detecting quest markers: {e}")
        
        return entities