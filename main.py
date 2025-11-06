"""
License Plate Detection System for Axis Security Camera
Detects moving vehicles and extracts license plate numbers
"""

import cv2
import numpy as np
from datetime import datetime
import os
import requests
from requests.auth import HTTPDigestAuth
import easyocr

class LicensePlateDetector:
    def __init__(self, camera_url, username, password, output_dir="plates"):
        """
        Initialize the license plate detector
        
        Args:
            camera_url: RTSP URL of the Axis camera (e.g., 'rtsp://192.168.1.100/axis-media/media.amp')
            username: Camera username
            password: Camera password
            output_dir: Directory to save detected plates
        """
        self.camera_url = camera_url
        self.username = username
        self.password = password
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize OCR reader (supports multiple languages)
        print("Initializing OCR reader...")
        self.reader = easyocr.Reader(['en'])
        
        # Motion detection parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.min_contour_area = 5000  # Minimum area to consider as vehicle
        
        # Initialize video capture
        self.cap = None
        
    def connect_camera(self):
        """Connect to the Axis camera"""
        print(f"Connecting to camera: {self.camera_url}")
        self.cap = cv2.VideoCapture(self.camera_url)
        
        if not self.cap.isOpened():
            raise Exception("Failed to connect to camera")
        
        print("Camera connected successfully")
        
    def detect_motion(self, frame):
        """
        Detect motion in frame using background subtraction
        
        Returns:
            List of contours representing moving objects
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (labeled as 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        vehicle_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        return vehicle_contours
    
    def extract_license_plate(self, frame, contour):
        """
        Extract and read license plate from vehicle region
        
        Args:
            frame: Original frame
            contour: Contour of detected vehicle
            
        Returns:
            Tuple of (plate_text, confidence, plate_image)
        """
        # Get bounding box of vehicle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract vehicle region with some padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return None, 0, None
        
        # Preprocess for better OCR results
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Try OCR on the vehicle region
        results = self.reader.readtext(gray)
        
        # Filter for text that looks like license plates
        # Typically alphanumeric, 5-8 characters
        best_plate = None
        best_confidence = 0
        
        for (bbox, text, confidence) in results:
            # Clean text
            text = text.upper().replace(" ", "").replace("-", "")
            
            # Check if it looks like a license plate
            if (5 <= len(text) <= 8 and 
                confidence > 0.5 and 
                any(c.isdigit() for c in text) and
                any(c.isalpha() for c in text)):
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_plate = text
        
        return best_plate, best_confidence, vehicle_roi
    
    def save_detection(self, plate_text, confidence, plate_image):
        """Save detected plate information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{plate_text}_{confidence:.2f}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, plate_image)
        
        # Also log to text file
        log_file = os.path.join(self.output_dir, "detections.log")
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{plate_text},{confidence:.2f},{filename}\n")
        
        print(f"Detected plate: {plate_text} (confidence: {confidence:.2f})")
    
    def run(self, display=True):
        """
        Main detection loop
        
        Args:
            display: Whether to display video feed (set False for headless operation)
        """
        self.connect_camera()
        
        print("Starting license plate detection... Press 'q' to quit")
        
        frame_skip = 2  # Process every Nth frame for efficiency
        frame_count = 0
        detected_plates = {}  # Track recently detected plates to avoid duplicates
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Skip frames for efficiency
                if frame_count % frame_skip != 0:
                    continue
                
                # Detect motion
                vehicle_contours = self.detect_motion(frame)
                
                # Process each detected vehicle
                for contour in vehicle_contours:
                    plate_text, confidence, plate_image = self.extract_license_plate(frame, contour)
                    
                    if plate_text and confidence > 0.6:
                        # Check if we've recently detected this plate (within 30 seconds)
                        current_time = datetime.now()
                        
                        if plate_text not in detected_plates or \
                           (current_time - detected_plates[plate_text]).seconds > 30:
                            
                            self.save_detection(plate_text, confidence, plate_image)
                            detected_plates[plate_text] = current_time
                    
                    # Draw bounding box on display
                    if display:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        if plate_text:
                            cv2.putText(frame, f"{plate_text} ({confidence:.2f})", 
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (0, 255, 0), 2)
                
                # Display frame
                if display:
                    cv2.imshow('License Plate Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        finally:
            self.cap.release()
            if display:
                cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Configure your Axis camera details
    # RTSP URL format: rtsp://<camera_ip>/axis-media/media.amp
    CAMERA_URL = "rtsp://192.168.1.100/axis-media/media.amp"
    USERNAME = "your_username"
    PASSWORD = "your_password"
    
    # Create detector instance
    detector = LicensePlateDetector(
        camera_url=CAMERA_URL,
        username=USERNAME,
        password=PASSWORD,
        output_dir="detected_plates"
    )
    
    # Run detection
    detector.run(display=True)
