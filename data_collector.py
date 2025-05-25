import cv2
import os
import time
from datetime import datetime

class DataCollector:
    def __init__(self, base_path="dataset"):
        self.base_path = base_path
        self.classes = {
            '1': 'normal',
            '2': 'looking_away', 
            '3': 'multiple_people',
            '4': 'phone_use',
            '5': 'talking',
            '6': 'no_person'
        }
        
        # Create directories
        self.create_directories()
        
        # Capture settings
        self.current_class = None
        self.auto_capture = False
        self.capture_interval = 0.5  # seconds
        self.last_capture_time = 0
        
    def create_directories(self):
        """Create dataset directories"""
        for class_name in self.classes.values():
            class_path = os.path.join(self.base_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            print(f"Created directory: {class_path}")
    
    def get_next_filename(self, class_name):
        """Get next available filename for a class"""
        class_path = os.path.join(self.base_path, class_name)
        existing_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        next_number = len(existing_files) + 1
        return os.path.join(class_path, f"{class_name}_{next_number:04d}.jpg")
    
    def save_image(self, frame, class_name):
        """Save image to appropriate class folder"""
        filename = self.get_next_filename(class_name)
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        return filename
    
    def draw_interface(self, frame):
        """Draw user interface on frame"""
        h, w = frame.shape[:2]
        
        # Draw instructions
        instructions = [
            "Data Collection for Proctoring System",
            "Press keys 1-6 to select class:",
            "1: Normal behavior",
            "2: Looking away",
            "3: Multiple people", 
            "4: Phone use",
            "5: Talking",
            "6: No person",
            "",
            "Controls:",
            "SPACE: Capture single image",
            "A: Toggle auto-capture",
            "Q: Quit"
        ]
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 350), (0, 0, 0), -1)