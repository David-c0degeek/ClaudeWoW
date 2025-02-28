"""
Image Processing Utilities

This module provides various image processing functions used throughout the ClaudeWoW project.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


def preprocess_image(image: np.ndarray, 
                     resize: Optional[Tuple[int, int]] = None,
                     grayscale: bool = False,
                     normalize: bool = False,
                     threshold: bool = False,
                     threshold_value: int = 127) -> np.ndarray:
    """
    Preprocesses an image for computer vision tasks
    
    Args:
        image: Input image as numpy array
        resize: Optional tuple of (width, height) to resize image to
        grayscale: Whether to convert image to grayscale
        normalize: Whether to normalize pixel values to range [0, 1]
        threshold: Whether to apply binary thresholding
        threshold_value: Threshold value if thresholding is applied
        
    Returns:
        Preprocessed image
    """
    # Create a copy to avoid modifying the original
    processed = image.copy()
    
    # Resize if needed
    if resize is not None:
        processed = cv2.resize(processed, resize)
    
    # Convert to grayscale if requested
    if grayscale:
        if len(processed.shape) == 3:  # Check if image is not already grayscale
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding if requested
    if threshold:
        if len(processed.shape) == 3:  # Ensure image is grayscale
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        _, processed = cv2.threshold(processed, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Normalize if requested
    if normalize:
        processed = processed.astype(np.float32) / 255.0
        
    return processed


def apply_ocr_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Applies preprocessing specifically for OCR text extraction
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image optimized for OCR
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply slight Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Apply morphological operations to clean up the text
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Invert back to black text on white background for OCR
    result = cv2.bitwise_not(opened)
    
    return result


def detect_edges(image: np.ndarray, 
                 low_threshold: int = 50, 
                 high_threshold: int = 150) -> np.ndarray:
    """
    Detects edges in an image using Canny edge detection
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Higher threshold for edge detection
        
    Returns:
        Edge detection result
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


def extract_color_range(image: np.ndarray, 
                        lower_bound: Tuple[int, int, int],
                        upper_bound: Tuple[int, int, int]) -> np.ndarray:
    """
    Extracts pixels within a specific color range (in HSV color space)
    
    Args:
        image: Input image in BGR format
        lower_bound: Lower HSV bound as (h, s, v)
        upper_bound: Upper HSV bound as (h, s, v)
        
    Returns:
        Binary mask where pixels within range are white
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create bounds as numpy arrays
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask


def find_contours(image: np.ndarray,
                  external_only: bool = True) -> List[np.ndarray]:
    """
    Finds contours in a binary image
    
    Args:
        image: Input binary image
        external_only: If True, only returns external contours
        
    Returns:
        List of contours
    """
    # Ensure image is binary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = image.copy()
    
    # Find contours
    mode = cv2.RETR_EXTERNAL if external_only else cv2.RETR_LIST
    contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def draw_bounding_boxes(image: np.ndarray,
                        boxes: List[Tuple[int, int, int, int]],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draws bounding boxes on an image
    
    Args:
        image: Input image
        boxes: List of bounding boxes as (x, y, width, height)
        color: RGB color for the boxes
        thickness: Line thickness
        
    Returns:
        Image with bounding boxes drawn
    """
    result = image.copy()
    
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    return result


def template_matching(image: np.ndarray, 
                      template: np.ndarray,
                      threshold: float = 0.8,
                      method: int = cv2.TM_CCOEFF_NORMED) -> List[Tuple[int, int, int, int]]:
    """
    Performs template matching to find occurrences of a template in an image
    
    Args:
        image: Image to search in
        template: Template to search for
        threshold: Minimum match threshold (0-1)
        method: OpenCV template matching method
        
    Returns:
        List of matches as (x, y, w, h)
    """
    # Get template dimensions
    h, w = template.shape[:2]
    
    # Perform matching
    result = cv2.matchTemplate(image, template, method)
    
    # Find locations where result exceeds threshold
    locations = np.where(result >= threshold)
    
    # Convert to list of rectangles
    matches = []
    for pt in zip(*locations[::-1]):  # Switch columns and rows
        matches.append((pt[0], pt[1], w, h))
    
    # Apply non-maximum suppression
    if len(matches) > 1:
        matches = non_max_suppression(matches, 0.3)
    
    return matches


def non_max_suppression(boxes: List[Tuple[int, int, int, int]], 
                        overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    Applies non-maximum suppression to remove overlapping boxes
    
    Args:
        boxes: List of bounding boxes as (x, y, width, height)
        overlap_threshold: Maximum allowed overlap ratio
        
    Returns:
        Filtered list of boxes
    """
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []
    
    # Convert to numpy array for easier calculations
    boxes_array = np.array(boxes)
    
    # Extract coordinates
    x = boxes_array[:, 0]
    y = boxes_array[:, 1]
    w = boxes_array[:, 2]
    h = boxes_array[:, 3]
    
    # Compute areas and indices
    areas = w * h
    indices = np.argsort(y + h)  # Sort by bottom-most edge
    
    keep = []
    
    while len(indices) > 0:
        # Get index of current box and add to keep list
        current = indices[-1]
        keep.append(current)
        indices = indices[:-1]
        
        # Skip if no indices left
        if len(indices) == 0:
            break
            
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x[current], x[indices])
        yy1 = np.maximum(y[current], y[indices])
        xx2 = np.minimum(x[current] + w[current], x[indices] + w[indices])
        yy2 = np.minimum(y[current] + h[current], y[indices] + h[indices])
        
        # Width and height of overlap area
        w_overlap = np.maximum(0, xx2 - xx1)
        h_overlap = np.maximum(0, yy2 - yy1)
        
        # Area of overlap
        overlap_area = w_overlap * h_overlap
        
        # Calculate IoU
        iou = overlap_area / (areas[current] + areas[indices] - overlap_area)
        
        # Keep indices with IoU below threshold
        indices = indices[iou <= overlap_threshold]
    
    # Return kept boxes
    return [boxes[i] for i in keep]