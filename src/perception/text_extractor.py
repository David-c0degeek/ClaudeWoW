"""
Text Extractor Module

This module handles OCR (Optical Character Recognition) to extract text from the game UI.
"""

import logging
import numpy as np
import cv2
import pytesseract
from typing import Dict, List, Tuple, Any, Optional
import re
import os
import time

class TextExtractor:
    """
    Extracts and processes text from World of Warcraft UI elements using OCR
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the TextExtractor
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.perception.text_extractor")
        self.config = config
        
        # Set pytesseract path if provided in config
        if "tesseract_path" in config:
            pytesseract.pytesseract.tesseract_cmd = config["tesseract_path"]
        
        # OCR configuration
        self.psm_mode = config.get("tesseract_psm", 7)  # Page segmentation mode
        self.tesseract_config = f"--psm {self.psm_mode} --oem 3"
        
        # Debug mode for saving processed images
        self.debug_mode = config.get("ocr_debug", False)
        self.debug_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "ocr_debug"
        )
        if self.debug_mode and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # Text cache to avoid repeated OCR on unchanged UI elements
        self.text_cache = {}
        self.cache_ttl = config.get("ocr_cache_ttl", 5)  # Cache time-to-live in seconds
        
        self.logger.info("TextExtractor initialized")
    
    def extract_text(self, screenshot: np.ndarray, rect: Tuple[int, int, int, int], 
                     multiline: bool = False, preprocess: bool = True,
                     cache_key: Optional[str] = None) -> str:
        """
        Extract text from a specific area of the screenshot
        
        Args:
            screenshot: The game screenshot
            rect: Rectangle coordinates (x, y, width, height)
            multiline: Whether to expect multiple lines of text
            preprocess: Whether to preprocess the image for better OCR
            cache_key: Key for caching results (if None, generates based on rect)
        
        Returns:
            str: Extracted text
        """
        try:
            x, y, w, h = rect
            
            # Check if the coordinates are valid
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > screenshot.shape[1] or y + h > screenshot.shape[0]:
                self.logger.warning(f"Invalid text extraction rectangle: {rect}")
                return ""
            
            # Create a cache key if not provided
            if cache_key is None:
                cache_key = f"{x}_{y}_{w}_{h}"
            
            # Check cache
            if cache_key in self.text_cache:
                entry = self.text_cache[cache_key]
                if time.time() - entry["timestamp"] < self.cache_ttl:
                    return entry["text"]
            
            # Extract the region of interest
            roi = screenshot[y:y+h, x:x+w].copy()
            
            # Preprocess the image if requested
            if preprocess:
                roi = self._preprocess_for_ocr(roi, multiline)
            
            # Save debug image if in debug mode
            if self.debug_mode:
                timestamp = int(time.time())
                debug_path = os.path.join(self.debug_dir, f"ocr_{cache_key}_{timestamp}.png")
                cv2.imwrite(debug_path, roi)
            
            # Set OCR parameters based on multiline flag
            config = self.tesseract_config
            if not multiline:
                config = "--psm 7 --oem 3"  # Single line mode
            else:
                config = "--psm 6 --oem 3"  # Multiline mode
            
            # Perform OCR
            text = pytesseract.image_to_string(roi, config=config).strip()
            
            # Clean the text
            text = self._clean_text(text)
            
            # Cache the result
            self.text_cache[cache_key] = {
                "text": text,
                "timestamp": time.time()
            }
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return ""
    
    def _preprocess_for_ocr(self, image: np.ndarray, multiline: bool = False) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Image to preprocess
            multiline: Whether the text is multiline
        
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize for better OCR if the image is too small
        if gray.shape[0] < 30:
            scale_factor = 30 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        return gray
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Text to clean
        
        Returns:
            str: Cleaned text
        """
        # Remove common OCR errors
        text = re.sub(r'[|]', 'I', text)  # Replace | with I
        text = re.sub(r'[\\_]', '/', text)  # Replace _ with /
        text = re.sub(r'[^a-zA-Z0-9\s\.,:\-\'/\(\)\[\]%]', '', text)  # Remove non-alphanumeric and common punctuation
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_number(self, screenshot: np.ndarray, rect: Tuple[int, int, int, int],
                      cache_key: Optional[str] = None) -> Optional[float]:
        """
        Extract a number from a specific area of the screenshot
        
        Args:
            screenshot: The game screenshot
            rect: Rectangle coordinates (x, y, width, height)
            cache_key: Key for caching results
        
        Returns:
            Optional[float]: Extracted number or None if not found
        """
        text = self.extract_text(screenshot, rect, multiline=False, cache_key=cache_key)
        
        # Extract numbers using regex
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        
        return None
    
    def extract_player_name(self, screenshot: np.ndarray, rect: Tuple[int, int, int, int]) -> str:
        """
        Extract player name with specialized preprocessing
        
        Args:
            screenshot: The game screenshot
            rect: Rectangle coordinates (x, y, width, height)
        
        Returns:
            str: Extracted player name
        """
        # Similar to extract_text but with specific optimizations for player names
        return self.extract_text(screenshot, rect, multiline=False, cache_key="player_name")
    
    def extract_quest_text(self, screenshot: np.ndarray, rect: Tuple[int, int, int, int]) -> Dict:
        """
        Extract and parse quest text
        
        Args:
            screenshot: The game screenshot
            rect: Rectangle coordinates (x, y, width, height)
        
        Returns:
            Dict: Structured quest information
        """
        text = self.extract_text(screenshot, rect, multiline=True, cache_key="quest_text")
        
        # Parse quest text
        lines = text.strip().split('\n')
        
        quest_info = {
            "title": "",
            "description": [],
            "objectives": []
        }
        
        if lines:
            quest_info["title"] = lines[0]
            
            in_description = True
            for line in lines[1:]:
                if "objectives:" in line.lower() or "objective:" in line.lower():
                    in_description = False
                    continue
                
                if in_description:
                    quest_info["description"].append(line)
                else:
                    # Try to parse objectives
                    if ":" in line or "/" in line:
                        quest_info["objectives"].append(line)
        
        return quest_info