"""
Image Processing Utilities

This module provides utility functions for image processing and analysis.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger("wow_ai.utils.image_processing")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for general computer vision tasks
    
    Args:
        image: Input image
    
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image

def apply_ocr_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image specifically for OCR (Optical Character Recognition)
    
    Args:
        image: Input image
    
    Returns:
        np.ndarray: Preprocessed image ready for OCR
    """
    try:
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
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    except Exception as e:
        logger.error(f"Error preprocessing image for OCR: {e}")
        return image

def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in an image using Canny edge detector
    
    Args:
        image: Input image
    
    Returns:
        np.ndarray: Edge map
    """
    try:
        # Preprocess the image
        preprocessed = preprocess_image(image)
        
        # Apply Canny edge detection
        edges = cv2.Canny(preprocessed, 50, 150)
        
        return edges
    except Exception as e:
        logger.error(f"Error detecting edges: {e}")
        return np.zeros_like(image)

def find_contours(image: np.ndarray) -> List:
    """
    Find contours in an image
    
    Args:
        image: Input image
    
    Returns:
        List: List of contours
    """
    try:
        # Ensure image is properly processed for contour detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    except Exception as e:
        logger.error(f"Error finding contours: {e}")
        return []

def match_template(image: np.ndarray, template: np.ndarray, threshold: float = 0.8) -> Optional[Tuple[int, int, int, int]]:
    """
    Match a template in an image
    
    Args:
        image: Image to search in
        template: Template to match
        threshold: Minimum match confidence (0.0 to 1.0)
    
    Returns:
        Optional[Tuple[int, int, int, int]]: Rectangle coordinates of match (x, y, width, height) or None if not found
    """
    try:
        # Convert to grayscale for template matching
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template.copy()
        
        # Get template dimensions
        h, w = gray_template.shape
        
        # Perform template matching
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # Find the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check if the match is good enough
        if max_val >= threshold:
            x, y = max_loc
            return (x, y, w, h)
        else:
            return None
    except Exception as e:
        logger.error(f"Error matching template: {e}")
        return None

def detect_color_range(image: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) -> np.ndarray:
    """
    Detect pixels within a specific color range
    
    Args:
        image: Input image
        lower_color: Lower bound of color range in BGR
        upper_color: Upper bound of color range in BGR
    
    Returns:
        np.ndarray: Binary mask of detected pixels
    """
    try:
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the specified color range
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        return mask
    except Exception as e:
        logger.error(f"Error detecting color range: {e}")
        return np.zeros_like(image[:,:,0])

def calculate_histogram(image: np.ndarray, mask: Optional[np.ndarray] = None) -> List:
    """
    Calculate color histograms for an image
    
    Args:
        image: Input image
        mask: Optional mask to limit histogram calculation
    
    Returns:
        List: List of histograms for each color channel
    """
    try:
        # Calculate histograms for each channel
        histograms = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
            histograms.append(hist)
        
        return histograms
    except Exception as e:
        logger.error(f"Error calculating histogram: {e}")
        return []

def detect_circles(image: np.ndarray, min_radius: int = 10, max_radius: int = 100) -> List[Tuple[int, int, int]]:
    """
    Detect circles in an image using Hough Circle Transform
    
    Args:
        image: Input image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
    
    Returns:
        List[Tuple[int, int, int]]: List of detected circles (x, y, radius)
    """
    try:
        # Preprocess the image
        gray = preprocess_image(image)
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        # Format results
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            return [(circle[0], circle[1], circle[2]) for circle in circles]
        else:
            return []
    except Exception as e:
        logger.error(f"Error detecting circles: {e}")
        return []

def calculate_bar_percentage(image: np.ndarray, expect_horizontal: bool = True) -> float:
    """
    Calculate the fill percentage of a bar (health, mana, etc.)
    
    Args:
        image: Image of the bar
        expect_horizontal: Whether the bar is horizontal (True) or vertical (False)
    
    Returns:
        float: Percentage of the bar that is filled (0.0 to 100.0)
    """
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for non-black pixels
        lower_color = np.array([0, 40, 40])
        upper_color = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Count non-zero (filled) pixels
        filled_pixels = cv2.countNonZero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        
        if total_pixels == 0:
            return 0.0
        
        # Calculate percentage
        if expect_horizontal:
            # For horizontal bars, calculate column-wise
            columns = np.sum(mask > 0, axis=0)
            non_zero_columns = np.count_nonzero(columns)
            if image.shape[1] == 0:
                return 0.0
            percentage = (non_zero_columns / image.shape[1]) * 100.0
        else:
            # For vertical bars, calculate row-wise
            rows = np.sum(mask > 0, axis=1)
            non_zero_rows = np.count_nonzero(rows)
            if image.shape[0] == 0:
                return 0.0
            percentage = (non_zero_rows / image.shape[0]) * 100.0
        
        return percentage
    except Exception as e:
        logger.error(f"Error calculating bar percentage: {e}")
        return 0.0

def detect_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions that might contain text
    
    Args:
        image: Input image
    
    Returns:
        List[Tuple[int, int, int, int]]: List of rectangle coordinates (x, y, width, height)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find text regions
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on aspect ratio and size
            aspect_ratio = w / h if h > 0 else 0
            if 0.1 < aspect_ratio < 10 and w > 10 and h > 5:
                text_regions.append((x, y, w, h))
        
        return text_regions
    except Exception as e:
        logger.error(f"Error detecting text regions: {e}")
        return []