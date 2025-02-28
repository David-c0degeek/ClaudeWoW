"""
Imitation Learning System

This module implements imitation learning algorithms for the AI agent
to learn strategies by observing and mimicking human gameplay.
"""

import logging
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
import cv2
import time
import json

# Setup module-level logger
logger = logging.getLogger("wow_ai.learning.imitation_learning")

class GameplayRecording:
    """
    Stores and processes recorded gameplay data for imitation learning
    """
    
    def __init__(self, recording_id: str, player_class: str, player_level: int):
        """
        Initialize a gameplay recording
        
        Args:
            recording_id: Unique identifier for this recording
            player_class: Character class in the recording
            player_level: Character level in the recording
        """
        self.recording_id = recording_id
        self.player_class = player_class
        self.player_level = player_level
        self.frames = []
        self.actions = []
        self.states = []
        self.timestamps = []
        self.metadata = {
            "recording_id": recording_id,
            "player_class": player_class,
            "player_level": player_level,
            "date_recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames": 0,
            "duration_seconds": 0,
            "tags": []
        }
    
    def add_frame(self, 
                 frame: np.ndarray, 
                 action: int, 
                 state: Dict[str, Any], 
                 timestamp: float) -> None:
        """
        Add a gameplay frame to the recording
        
        Args:
            frame: Screenshot of the game
            action: Action taken by the player
            state: Game state at this frame
            timestamp: Time when frame was captured
        """
        self.frames.append(frame)
        self.actions.append(action)
        self.states.append(state)
        self.timestamps.append(timestamp)
        
        # Update metadata
        self.metadata["total_frames"] = len(self.frames)
        if len(self.timestamps) > 1:
            self.metadata["duration_seconds"] = self.timestamps[-1] - self.timestamps[0]
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to describe this recording
        
        Args:
            tag: Descriptive tag (e.g., "raid", "pvp", "leveling")
        """
        if tag not in self.metadata["tags"]:
            self.metadata["tags"].append(tag)
    
    def save(self, base_path: str) -> None:
        """
        Save recording to disk
        
        Args:
            base_path: Base directory to save recording
        """
        recording_dir = os.path.join(base_path, self.recording_id)
        os.makedirs(recording_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(recording_dir, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save frames, actions, states
        frames_path = os.path.join(recording_dir, "frames.npy")
        actions_path = os.path.join(recording_dir, "actions.npy")
        states_path = os.path.join(recording_dir, "states.pkl")
        timestamps_path = os.path.join(recording_dir, "timestamps.npy")
        
        np.save(frames_path, np.array(self.frames))
        np.save(actions_path, np.array(self.actions))
        np.save(timestamps_path, np.array(self.timestamps))
        
        with open(states_path, 'wb') as f:
            pickle.dump(self.states, f)
            
        logger.info(f"Saved recording {self.recording_id} with {len(self.frames)} frames")
    
    @classmethod
    def load(cls, recording_id: str, base_path: str) -> 'GameplayRecording':
        """
        Load recording from disk
        
        Args:
            recording_id: Recording identifier
            base_path: Base directory containing recordings
            
        Returns:
            Loaded GameplayRecording object
        """
        recording_dir = os.path.join(base_path, recording_id)
        
        # Load metadata
        with open(os.path.join(recording_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Create recording object
        recording = cls(
            recording_id=metadata["recording_id"],
            player_class=metadata["player_class"],
            player_level=metadata["player_level"]
        )
        
        # Load data
        recording.frames = np.load(os.path.join(recording_dir, "frames.npy")).tolist()
        recording.actions = np.load(os.path.join(recording_dir, "actions.npy")).tolist()
        recording.timestamps = np.load(os.path.join(recording_dir, "timestamps.npy")).tolist()
        
        with open(os.path.join(recording_dir, "states.pkl"), 'rb') as f:
            recording.states = pickle.load(f)
        
        # Update metadata
        recording.metadata = metadata
        
        logger.info(f"Loaded recording {recording_id} with {len(recording.frames)} frames")
        return recording


class GameplayDataset(Dataset):
    """
    PyTorch dataset for gameplay data
    """
    
    def __init__(self, recordings: List[GameplayRecording], transform=None):
        """
        Initialize dataset from gameplay recordings
        
        Args:
            recordings: List of gameplay recordings
            transform: Optional transform to apply to frames
        """
        self.frames = []
        self.actions = []
        self.states = []
        
        # Combine all recordings
        for recording in recordings:
            self.frames.extend(recording.frames)
            self.actions.extend(recording.actions)
            self.states.extend(recording.states)
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self.frames[idx]
        action = self.actions[idx]
        state = self.states[idx]
        
        if self.transform:
            frame = self.transform(frame)
        
        return {
            "frame": frame,
            "action": action,
            "state": state
        }


class BehavioralCloning(nn.Module):
    """
    Neural network for behavioral cloning
    Learns to predict player actions from game state
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """
        Initialize behavioral cloning model
        
        Args:
            input_dim: Dimension of state vector
            output_dim: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (state vector)
            
        Returns:
            Action logits
        """
        return self.model(x)


class VisualBehavioralCloning(nn.Module):
    """
    Neural network for behavioral cloning from visual input
    Learns to predict player actions from game screenshots
    """
    
    def __init__(self, num_actions: int):
        """
        Initialize visual behavioral cloning model
        
        Args:
            num_actions: Number of possible actions
        """
        super().__init__()
        
        # CNN for processing images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the flattened size (depends on input dimensions)
        self._calc_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def _calc_conv_output_size(self) -> None:
        """Calculate the output size of the convolutional layers"""
        # Assume standard game resolution
        x = torch.zeros(1, 3, 480, 640)
        x = self.conv_layers(x)
        self.conv_output_size = x.size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (game screenshot)
            
        Returns:
            Action logits
        """
        x = self.conv_layers(x)
        return self.fc_layers(x)


class ImitationLearningManager:
    """Manager class for imitation learning"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the imitation learning manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = logging.getLogger("wow_ai.learning.imitation_learning")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.recordings_dir = os.path.join(self.base_dir, "data", "recordings")
        self.models_dir = os.path.join(self.base_dir, "data", "models", "learning")
        
        # Ensure directories exist
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Configure models
        self.num_actions = config.get("learning", {}).get("num_actions", 20)
        self.state_dim = config.get("learning", {}).get("state_dim", 100)
        
        # Create models
        self.bc_model = BehavioralCloning(
            input_dim=self.state_dim,
            output_dim=self.num_actions
        ).to(self.device)
        
        self.visual_bc_model = VisualBehavioralCloning(
            num_actions=self.num_actions
        ).to(self.device)
        
        # Optimizers
        self.bc_optimizer = optim.Adam(
            self.bc_model.parameters(),
            lr=config.get("learning", {}).get("bc_learning_rate", 1e-4)
        )
        
        self.visual_bc_optimizer = optim.Adam(
            self.visual_bc_model.parameters(),
            lr=config.get("learning", {}).get("visual_bc_learning_rate", 1e-4)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Load saved models if available
        self._load_models()
    
    def _load_models(self) -> None:
        """Load saved models if available"""
        bc_path = os.path.join(self.models_dir, "behavioral_cloning.pt")
        visual_bc_path = os.path.join(self.models_dir, "visual_behavioral_cloning.pt")
        
        if os.path.exists(bc_path):
            try:
                self.bc_model.load_state_dict(torch.load(bc_path, map_location=self.device))
                logger.info("Loaded behavioral cloning model")
            except Exception as e:
                logger.error(f"Failed to load behavioral cloning model: {e}")
        
        if os.path.exists(visual_bc_path):
            try:
                self.visual_bc_model.load_state_dict(torch.load(visual_bc_path, map_location=self.device))
                logger.info("Loaded visual behavioral cloning model")
            except Exception as e:
                logger.error(f"Failed to load visual behavioral cloning model: {e}")
    
    def save_models(self) -> None:
        """Save current models"""
        bc_path = os.path.join(self.models_dir, "behavioral_cloning.pt")
        visual_bc_path = os.path.join(self.models_dir, "visual_behavioral_cloning.pt")
        
        torch.save(self.bc_model.state_dict(), bc_path)
        torch.save(self.visual_bc_model.state_dict(), visual_bc_path)
        
        logger.info("Saved imitation learning models")
    
    def start_recording(self, player_class: str, player_level: int) -> GameplayRecording:
        """
        Start recording gameplay
        
        Args:
            player_class: Character class
            player_level: Character level
            
        Returns:
            New GameplayRecording object
        """
        recording_id = f"{player_class.lower()}_{player_level}_{int(time.time())}"
        recording = GameplayRecording(recording_id, player_class, player_level)
        return recording
    
    def get_available_recordings(self) -> List[str]:
        """
        Get list of available recordings
        
        Returns:
            List of recording IDs
        """
        try:
            return [d for d in os.listdir(self.recordings_dir) 
                   if os.path.isdir(os.path.join(self.recordings_dir, d))]
        except FileNotFoundError:
            return []
    
    def load_recordings(self, recording_ids: Optional[List[str]] = None) -> List[GameplayRecording]:
        """
        Load specified recordings
        
        Args:
            recording_ids: List of recording IDs to load, or None to load all
            
        Returns:
            List of loaded GameplayRecording objects
        """
        if recording_ids is None:
            recording_ids = self.get_available_recordings()
        
        recordings = []
        for rec_id in recording_ids:
            try:
                recording = GameplayRecording.load(rec_id, self.recordings_dir)
                recordings.append(recording)
            except Exception as e:
                logger.error(f"Failed to load recording {rec_id}: {e}")
        
        return recordings
    
    def prepare_state_dataset(self, recordings: List[GameplayRecording]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare dataset from game states
        
        Args:
            recordings: List of gameplay recordings
            
        Returns:
            Tuple of (states, actions) tensors
        """
        states = []
        actions = []
        
        for recording in recordings:
            for i in range(len(recording.states)):
                # Convert state dict to vector
                state_vector = self._state_to_vector(recording.states[i])
                if state_vector is not None:
                    states.append(state_vector)
                    actions.append(recording.actions[i])
        
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        
        return states_tensor, actions_tensor
    
    def _transform_image(self, image):
        """
        Transform image for visual model training
        Defined as a method to make it pickle-able for multiprocessing
        
        Args:
            image: Raw image
            
        Returns:
            Transformed tensor
        """
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            self.logger.warning(f"Image is not a numpy array, type: {type(image)}")
            # If not an ndarray, create a small black image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            # Resize to standard size
            image = cv2.resize(image, (640, 480))
            # Convert to torch tensor and normalize
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return image
        except Exception as e:
            self.logger.error(f"Error transforming image: {e}")
            # Return a zero tensor as fallback
            return torch.zeros((3, 480, 640), dtype=torch.float32)
    
    def prepare_visual_dataset(self, recordings: List[GameplayRecording]) -> DataLoader:
        """
        Prepare DataLoader for visual training
        
        Args:
            recordings: List of gameplay recordings
            
        Returns:
            DataLoader for visual training
        """
        # Create dataset
        dataset = GameplayDataset(recordings, transform=self._transform_image)
        
        # Create dataloader
        batch_size = self.config.get("learning", {}).get("batch_size", 32)
        
        # Use zero workers for simpler debugging or if multiprocessing is problematic
        num_workers = self.config.get("learning", {}).get("num_workers", 2)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
        
        return dataloader
    
    def _state_to_vector(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert state dictionary to vector
        
        Args:
            state: Game state dictionary
            
        Returns:
            State vector or None if conversion fails
        """
        # This is a placeholder - actual implementation depends on game state structure
        try:
            vector = np.zeros(self.state_dim, dtype=np.float32)
            
            # Example features - replace with actual state fields
            if "health_percent" in state:
                vector[0] = state["health_percent"]
            if "mana_percent" in state:
                vector[1] = state["mana_percent"]
            if "target_health_percent" in state:
                vector[2] = state["target_health_percent"]
            if "distance_to_target" in state:
                vector[3] = min(state.get("distance_to_target", 0) / 100.0, 1.0)
            
            # Add cooldown information
            cooldowns = state.get("cooldowns", {})
            for i, (ability, cd) in enumerate(cooldowns.items()):
                if 4 + i < self.state_dim:
                    vector[4 + i] = min(cd / 300.0, 1.0)  # Normalize cooldown
            
            return vector
        except Exception as e:
            logger.error(f"Failed to convert state to vector: {e}")
            return None
    
    def train_from_states(self, recordings: List[GameplayRecording], epochs: int = 10) -> None:
        """
        Train behavioral cloning model from game states
        
        Args:
            recordings: List of gameplay recordings
            epochs: Number of training epochs
        """
        # Prepare dataset
        states, actions = self.prepare_state_dataset(recordings)
        
        if len(states) == 0:
            logger.warning("No valid state data for training")
            return
        
        logger.info(f"Training behavioral cloning on {len(states)} examples")
        
        # Training loop
        self.bc_model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.bc_model(states)
            loss = self.criterion(outputs, actions)
            
            # Backward and optimize
            self.bc_optimizer.zero_grad()
            loss.backward()
            self.bc_optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Save model
        self.save_models()
        
        # Evaluate
        self.bc_model.eval()
        with torch.no_grad():
            outputs = self.bc_model(states)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == actions).sum().item() / actions.size(0)
            logger.info(f"Training accuracy: {accuracy:.4f}")
    
    def train_from_visual(self, recordings: List[GameplayRecording], epochs: int = 10) -> None:
        """
        Train visual behavioral cloning model from game screenshots
        
        Args:
            recordings: List of gameplay recordings
            epochs: Number of training epochs
        """
        # Prepare dataloader
        dataloader = self.prepare_visual_dataset(recordings)
        
        if len(dataloader) == 0:
            logger.warning("No valid visual data for training")
            return
        
        logger.info(f"Training visual behavioral cloning on {len(dataloader.dataset)} examples")
        
        # Training loop
        self.visual_bc_model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                frames = batch["frame"].to(self.device)
                actions = batch["action"].to(self.device)
                
                # Forward pass
                outputs = self.visual_bc_model(frames)
                loss = self.criterion(outputs, actions)
                
                # Backward and optimize
                self.visual_bc_optimizer.zero_grad()
                loss.backward()
                self.visual_bc_optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()
            
            # Print epoch stats
            if (epoch + 1) % 1 == 0:
                avg_loss = total_loss / len(dataloader)
                accuracy = correct / total
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save model
        self.save_models()
    
    def predict_action_from_state(self, state: Dict[str, Any]) -> int:
        """
        Predict action from game state
        
        Args:
            state: Current game state
            
        Returns:
            Predicted action
        """
        # Convert state to vector
        state_vector = self._state_to_vector(state)
        if state_vector is None:
            # Fallback to random action if state conversion fails
            return np.random.randint(0, self.num_actions)
        
        # Convert to tensor
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        self.bc_model.eval()
        with torch.no_grad():
            outputs = self.bc_model(state_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.item()
    
    def predict_action_from_frame(self, frame: np.ndarray) -> int:
        """
        Predict action from game screenshot
        
        Args:
            frame: Game screenshot
            
        Returns:
            Predicted action
        """
        try:
            # Ensure frame is a numpy array
            if not isinstance(frame, np.ndarray):
                self.logger.warning(f"Frame is not a numpy array, type: {type(frame)}")
                # Create a default black frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Preprocess frame
            frame = cv2.resize(frame, (640, 480))
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            
            # Predict
            self.visual_bc_model.eval()
            with torch.no_grad():
                outputs = self.visual_bc_model(frame_tensor)
                _, predicted = torch.max(outputs.data, 1)
                return predicted.item()
        except Exception as e:
            self.logger.error(f"Error predicting from frame: {e}")
            # Return a default action
            return 0
    
    def extract_optimal_rotation(self, 
                                recordings: List[GameplayRecording], 
                                class_name: str, 
                                specialization: str) -> Dict[str, Any]:
        """
        Extract optimal ability rotation from gameplay recordings
        
        Args:
            recordings: List of gameplay recordings
            class_name: Character class
            specialization: Character specialization
            
        Returns:
            Dictionary containing rotation information
        """
        # Filter recordings by class
        class_recordings = [r for r in recordings if r.player_class.lower() == class_name.lower()]
        
        if not class_recordings:
            logger.warning(f"No recordings found for class: {class_name}")
            return {"status": "error", "message": "No recordings found for class"}
        
        # Analyze action sequences
        action_sequences = []
        
        for recording in class_recordings:
            for i in range(len(recording.actions) - 5):
                sequence = recording.actions[i:i+5]
                action_sequences.append(sequence)
        
        # Find common patterns
        pattern_counts = {}
        
        for seq in action_sequences:
            key = tuple(seq)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        
        # Sort by frequency
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top patterns
        top_patterns = [{"sequence": list(pattern), "count": count} 
                       for pattern, count in sorted_patterns[:10]]
        
        result = {
            "class": class_name,
            "specialization": specialization,
            "total_sequences_analyzed": len(action_sequences),
            "top_patterns": top_patterns,
            "status": "success"
        }
        
        return result
    
    def learn_navigation_shortcuts(self, recordings: List[GameplayRecording]) -> Dict[str, Any]:
        """
        Learn navigation shortcuts from gameplay recordings
        
        Args:
            recordings: List of gameplay recordings
            
        Returns:
            Dictionary containing navigation shortcuts
        """
        # Analyze movement patterns
        shortcuts = {}
        
        # Configuration for shortcut detection
        min_distance = 10.0  # Minimum distance to consider as a potential shortcut
        expected_speed = 5.0  # Expected units per frame for normal movement
        speed_threshold = 2.0  # How many times faster than expected to be a shortcut
        
        for recording in recordings:
            for i in range(len(recording.states) - 1):
                current_state = recording.states[i]
                next_state = recording.states[i+1]
                
                # Check if this is a navigation action
                action = recording.actions[i]
                
                # Get locations
                current_location = current_state.get("location", {})
                next_location = next_state.get("location", {})
                
                if not current_location or not next_location:
                    continue
                
                # Get position coordinates
                start_point = (
                    current_location.get("x", 0),
                    current_location.get("y", 0),
                    current_location.get("z", 0)
                )
                
                end_point = (
                    next_location.get("x", 0),
                    next_location.get("y", 0),
                    next_location.get("z", 0)
                )
                
                # Calculate distance
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                dz = end_point[2] - start_point[2]
                distance = (dx**2 + dy**2 + dz**2)**0.5
                
                # Skip if movement is too small
                if distance < min_distance:
                    continue
                
                # Calculate time between frames
                time_current = recording.timestamps[i]
                time_next = recording.timestamps[i+1]
                time_diff = max(0.001, time_next - time_current)  # Prevent division by zero
                
                # Calculate movement speed
                speed = distance / time_diff
                
                # Check if this is significantly faster than expected
                is_shortcut = speed > (expected_speed * speed_threshold)
                
                # Or check if there's a significant height/terrain change that suggests a jump
                height_change = abs(dz)
                is_vertical_shortcut = height_change > 5.0  # Significant height change
                
                if is_shortcut or is_vertical_shortcut:
                    zone = current_location.get("zone", "unknown")
                    start_key = f"{zone}_{int(start_point[0])}_{int(start_point[1])}"
                    end_key = f"{zone}_{int(end_point[0])}_{int(end_point[1])}"
                    
                    shortcut_key = f"{start_key}_to_{end_key}"
                    
                    if shortcut_key not in shortcuts:
                        shortcuts[shortcut_key] = {
                            "zone": zone,
                            "start": start_point,
                            "end": end_point,
                            "distance": distance,
                            "speed": speed,
                            "expected_speed": expected_speed,
                            "height_change": height_change,
                            "is_vertical": is_vertical_shortcut,
                            "count": 1,
                            "actions": [action],
                            "frame_indices": [i]
                        }
                    else:
                        shortcuts[shortcut_key]["count"] += 1
                        if action not in shortcuts[shortcut_key]["actions"]:
                            shortcuts[shortcut_key]["actions"].append(action)
                        shortcuts[shortcut_key]["frame_indices"].append(i)
        
        # Filter significant shortcuts with looser criteria for testing
        # In production, you might want to require more occurrences
        significant_shortcuts = {}
        for k, v in shortcuts.items():
            # Consider significant if:
            # 1. Used multiple times, OR
            # 2. Very fast movement, OR
            # 3. Significant vertical movement
            if (v["count"] > 1 or 
                v["speed"] > expected_speed * 3 or  
                v["height_change"] > 10.0):
                significant_shortcuts[k] = v
        
        # For debugging - add some computed values to describe why it's a shortcut
        for k, v in significant_shortcuts.items():
            v["reason"] = []
            if v["count"] > 1:
                v["reason"].append("multiple_uses")
            if v["speed"] > expected_speed * speed_threshold:
                v["reason"].append("fast_movement")
            if v["height_change"] > 5.0:
                v["reason"].append("vertical_shortcut")
        
        result = {
            "total_shortcuts_detected": len(significant_shortcuts),
            "shortcuts": list(significant_shortcuts.values()),
            "status": "success"
        }
        
        return result