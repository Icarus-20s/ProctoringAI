import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import json

# Same model architecture as training
class ProctoringCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ProctoringCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers with activation and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class RealTimeProctoringSystem:
    def __init__(self, model_path='best_proctoring_model.pth', device=None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Class names
        self.classes = ['normal', 'looking_away', 'multiple_people', 'phone_use', 'talking', 'no_person']
        
        # Load model
        self.model = ProctoringCNN(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Violation tracking
        self.violations = []
        self.prediction_history = []
        self.confidence_threshold = 0.7
        self.violation_persistence = 3  # frames to confirm violation
        
        # Colors for different classes
        self.colors = {
            'normal': (0, 255, 0),          # Green
            'looking_away': (0, 165, 255),  # Orange
            'multiple_people': (0, 0, 255), # Red
            'phone_use': (255, 0, 0),       # Blue
            'talking': (255, 255, 0),       # Cyan
            'no_person': (128, 128, 128)    # Gray
        }
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.classes}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        
        # Add batch dimension
        batch_tensor = tensor_image.unsqueeze(0)
        
        return batch_tensor.to(self.device)
    
    def predict(self, frame):
        """Make prediction on frame"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_frame(frame)
            
            # Forward pass
            outputs = self.model(input_tensor)
            
            # Get probabilities
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            return predicted_class.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def update_prediction_history(self, prediction, confidence):
        """Update prediction history for stability"""
        self.prediction_history.append((prediction, confidence))
        
        # Keep only recent predictions
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
    
    def get_stable_prediction(self):
        """Get stable prediction based on recent history"""
        if len(self.prediction_history) < 3:
            return None, 0.0
        
        # Get recent predictions
        recent_predictions = self.prediction_history[-5:]
        
        # Count predictions
        prediction_counts = {}
        total_confidence = 0
        
        for pred, conf in recent_predictions:
            if pred not in prediction_counts:
                prediction_counts[pred] = []
            prediction_counts[pred].append(conf)
            total_confidence += conf
        
        # Find most common prediction
        max_count = 0
        stable_prediction = None
        avg_confidence = 0
        
        for pred, confidences in prediction_counts.items():
            if len(confidences) > max_count:
                max_count = len(confidences)
                stable_prediction = pred
                avg_confidence = sum(confidences) / len(confidences)
        
        return stable_prediction, avg_confidence
    
    def check_violation(self, prediction, confidence):
        """Check if current prediction is a violation"""
        if prediction == 0:  # Normal behavior
            return False
        
        # Check confidence threshold
        if confidence > self.confidence_threshold:
            return True
        
        return False
    
    def log_violation(self, prediction, confidence):
        """Log violation with details"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        violation = {
            'time': timestamp,
            'type': self.classes[prediction],
            'confidence': round(confidence * 100, 2)
        }
        self.violations.append(violation)
        print(f"VIOLATION: {timestamp} - {self.classes[prediction]} ({confidence*100:.1f}% confidence)")
    
    def draw_predictions(self, frame, prediction, confidence, probabilities):
        """Draw predictions and information on frame"""
        h, w = frame.shape[:2]
        
        # Get predicted class name and color
        class_name = self.classes[prediction]
        color = self.colors[class_name]
        
        # Draw main prediction
        text = f"{class_name}: {confidence*100:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw violation warning
        if self.check_violation(prediction, confidence):
            cv2.putText(frame, "VIOLATION DETECTED!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw violation count
        cv2.putText(frame, f"Total Violations: {len(self.violations)}", 
                   (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw probability bar
        bar_width = 200
        bar_height = 20
        start_x = 10
        start_y = h - 150
        
        cv2.putText(frame, "Confidence:", (start_x, start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, (class_name, prob) in enumerate(zip(self.classes, probabilities)):
            y = start_y + i * 25
            
            # Draw background bar
            cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), 
                         (50, 50, 50), -1)
            
            # Draw probability bar
            prob_width = int(bar_width * prob)
            color = self.colors[class_name]
            cv2.rectangle(frame, (start_x, y), (start_x + prob_width, y + bar_height), 
                         color, -1)
            
            # Draw text
            text = f"{class_name}: {prob*100:.1f}%"
            cv2.putText(frame, text, (start_x + bar_width + 10, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_violation_report(self, filename='violation_report.json'):
        """Save violation report to file"""
        report = {
            'total_violations': len(self.violations),
            'violations': self.violations,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Violation report saved to {filename}")
    
    def run_realtime_monitoring(self):
        """Run real-time monitoring system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Real-time Proctoring System Started!")
        print("Press 'q' to quit")
        print("Press 's' to save violation report")
        print("Press 'r' to reset violations")
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make prediction
            prediction, confidence, probabilities = self.predict(frame)
            
            # Update history
            self.update_prediction_history(prediction, confidence)
            
            # Check for violations
            if self.check_violation(prediction, confidence):
                # Get stable prediction to avoid false positives
                stable_pred, stable_conf = self.get_stable_prediction()
                if stable_pred is not None and stable_pred != 0 and stable_conf > self.confidence_threshold:
                    # Check if this is a new violation (not repeated)
                    if len(self.violations) == 0 or self.violations[-1]['type'] != self.classes[stable_pred]:
                        self.log_violation(stable_pred, stable_conf)
            
            # Draw predictions
            frame = self.draw_predictions(frame, prediction, confidence, probabilities)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
                
            cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('AI Proctoring System - Custom Model', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_violation_report()
            elif key == ord('r'):
                self.violations = []
                print("Violations reset!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        print(f"\nFinal Report: {len(self.violations)} violations detected")
        self.save_violation_report()

def main():
    try:
        # Initialize the system
        proctor = RealTimeProctoringSystem()
        
        # Run real-time monitoring
        proctor.run_realtime_monitoring()
        
    except FileNotFoundError:
        print("Error: Model file 'best_proctoring_model.pth' not found!")
        print("Please train the model first using the training script.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()