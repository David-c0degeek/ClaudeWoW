"""
Test script for the imitation learning system
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.learning.imitation_learning import ImitationLearningManager, GameplayRecording
from src.utils.config import load_config

def create_dummy_game_data(num_frames=100):
    """Create dummy game data for testing"""
    frames = []
    actions = []
    states = []
    timestamps = []
    
    # Create dummy screen images (black frames with some random pixels)
    for i in range(num_frames):
        # Create a small 480x640 black frame with some random pixels
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some random pixels
        for _ in range(50):
            x = np.random.randint(0, 640)
            y = np.random.randint(0, 480)
            color = np.random.randint(0, 255, 3).tolist()
            frame[y, x] = color
        
        # Add some UI elements
        if i % 2 == 0:
            # Add health bar (red rectangle)
            cv2.rectangle(frame, (50, 50), (150, 60), (0, 0, 255), -1)
        else:
            # Add health bar with less health
            cv2.rectangle(frame, (50, 50), (100, 60), (0, 0, 255), -1)
        
        # Add mana bar (blue rectangle)
        cv2.rectangle(frame, (50, 70), (130, 80), (255, 0, 0), -1)
        
        # Simulate an enemy every 10 frames
        if i % 10 == 0:
            # Add enemy (green rectangle)
            cv2.rectangle(frame, (320, 240), (350, 270), (0, 255, 0), -1)
        
        frames.append(frame)
        
        # Create dummy actions (0-19 as per our specification)
        # Simulate common action sequences like: target, cast1, cast2, cast3
        if i % 10 == 0:
            action = 5  # target nearest enemy
        elif i % 10 == 1:
            action = 7  # cast spell 1
        elif i % 10 == 2:
            action = 8  # cast spell 2  
        elif i % 10 == 3:
            action = 9  # cast spell 3
        else:
            action = np.random.randint(0, 20)
        
        actions.append(action)
        
        # Create dummy game states
        state = {
            "health_percent": 0.8 if i % 2 == 0 else 0.7,
            "mana_percent": 0.9,
            "target_health_percent": 0.5 if i % 10 < 5 else 0.0,
            "distance_to_target": 10,
            "combat_state": i % 10 < 5,  # in combat for 5 frames, then not in combat
            "cooldowns": {
                "ability1": max(0, 3 - (i % 4)),
                "ability2": max(0, 5 - (i % 6)),
                "ability3": max(0, 10 - (i % 11))
            },
            "location": {
                "zone": "Elwynn Forest",
                "x": 100 + i * 0.5,
                "y": 200 + (i % 5),
                "z": 30
            }
        }
        
        # Every 15 frames, simulate a navigation shortcut
        if i % 15 == 0 and i > 0:
            # Big jump in coordinates to simulate a shortcut
            state["location"]["x"] += 20
            state["location"]["y"] += 15
            state["location"]["z"] += 5  # Add some height change
            
            # Make timestamps further apart to show this is a fast movement
            if i > 0:
                timestamps[-1] += 0.2  # Add delay before the shortcut
        
        states.append(state)
        timestamps.append(time.time() + i * 0.1)  # timestamps 0.1s apart
    
    return frames, actions, states, timestamps

def test_recording_functionality():
    """Test the gameplay recording functionality"""
    print("Testing gameplay recording functionality...")
    
    config = load_config()
    imitation_learning = ImitationLearningManager(config)
    
    # Create a new recording
    recording_id = f"test_warrior_{int(time.time())}"
    recording = GameplayRecording(recording_id, "Warrior", 60)
    
    # Add some dummy data
    frames, actions, states, timestamps = create_dummy_game_data(100)
    for i in range(len(frames)):
        recording.add_frame(frames[i], actions[i], states[i], timestamps[i])
    
    # Add some tags
    recording.add_tag("test")
    recording.add_tag("warrior")
    recording.add_tag("combat")
    
    # Save the recording
    recording.save(imitation_learning.recordings_dir)
    print(f"Recording saved with {len(recording.frames)} frames")
    
    # List available recordings
    recordings = imitation_learning.get_available_recordings()
    print(f"Available recordings: {recordings}")
    
    # Load the recording
    loaded_recordings = imitation_learning.load_recordings([recording_id])
    if loaded_recordings:
        loaded_recording = loaded_recordings[0]
        print(f"Loaded recording: {loaded_recording.recording_id}")
        print(f"Number of frames: {len(loaded_recording.frames)}")
        print(f"Player class: {loaded_recording.player_class}")
        print(f"Player level: {loaded_recording.player_level}")
        print(f"Tags: {loaded_recording.metadata['tags']}")
    
    return loaded_recordings[0] if loaded_recordings else None

def test_rotation_extraction(recording):
    """Test the rotation extraction functionality"""
    print("\nTesting rotation extraction...")
    
    config = load_config()
    imitation_learning = ImitationLearningManager(config)
    
    # Extract optimal rotations
    rotations = imitation_learning.extract_optimal_rotation(
        [recording], "Warrior", "Arms"
    )
    
    print("Extracted rotations:")
    print(json.dumps(rotations, indent=2))
    
    return rotations

def test_navigation_shortcuts(recording):
    """Test the navigation shortcuts learning"""
    print("\nTesting navigation shortcuts learning...")
    
    config = load_config()
    imitation_learning = ImitationLearningManager(config)
    
    # Learn navigation shortcuts
    shortcuts = imitation_learning.learn_navigation_shortcuts([recording])
    
    print("Learned navigation shortcuts:")
    print(json.dumps(shortcuts, indent=2))
    
    return shortcuts

def test_model_training(recording):
    """Test the model training functionality"""
    print("\nTesting model training...")
    
    config = load_config()
    imitation_learning = ImitationLearningManager(config)
    
    # Train from states
    print("Training behavioral cloning model from states...")
    imitation_learning.train_from_states([recording], epochs=3)
    
    # Train from visual data (with num_workers=0 to avoid pickling issues)
    print("Training visual behavioral cloning model...")
    # Force num_workers to 0 for testing
    old_num_workers = imitation_learning.config.get("learning", {}).get("num_workers", 2)
    if "learning" not in imitation_learning.config:
        imitation_learning.config["learning"] = {}
    imitation_learning.config["learning"]["num_workers"] = 0
    
    # Train the model
    imitation_learning.train_from_visual([recording], epochs=3)
    
    # Restore original config
    imitation_learning.config["learning"]["num_workers"] = old_num_workers
    
    # Test prediction
    print("Testing prediction from state...")
    action = imitation_learning.predict_action_from_state(recording.states[0])
    print(f"Predicted action: {action}")
    
    print("Testing prediction from frame...")
    action = imitation_learning.predict_action_from_frame(recording.frames[0])
    print(f"Predicted action: {action}")

def main():
    """Main test function"""
    print("Starting imitation learning tests...")
    
    # Test recording functionality
    recording = test_recording_functionality()
    
    if recording:
        # Test rotation extraction
        test_rotation_extraction(recording)
        
        # Test navigation shortcuts
        test_navigation_shortcuts(recording)
        
        # Test model training
        test_model_training(recording)
    
    print("\nAll tests completed")

if __name__ == "__main__":
    main()