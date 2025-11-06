"""
License Plate Detection System for Axis Security Camera
Detects moving vehicles, extracts license plates using OpenAI Vision API,
stores data in MongoDB, and uploads images to AWS S3
"""

import cv2
import numpy as np
from datetime import datetime
import os
import base64
from io import BytesIO
from PIL import Image
import requests
from pymongo import MongoClient
import boto3
from botocore.exceptions import ClientError
from openai import OpenAI

class LicensePlateDetector:
    def __init__(self, camera_url, username, password, 
                 openai_api_key, mongodb_uri, aws_access_key, 
                 aws_secret_key, s3_bucket_name, s3_region="us-east-1"):
        """
        Initialize the license plate detector
        
        Args:
            camera_url: RTSP URL of the Axis camera
            username: Camera username
            password: Camera password
            openai_api_key: OpenAI API key
            mongodb_uri: MongoDB connection string (e.g., 'mongodb://localhost:27017/')
            aws_access_key: AWS access key ID
            aws_secret_key: AWS secret access key
            s3_bucket_name: S3 bucket name for image storage
            s3_region: AWS region for S3 bucket
        """
        self.camera_url = camera_url
        self.username = username
        self.password = password
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize MongoDB
        print("Connecting to MongoDB...")
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client['license_plate_db']
        self.collection = self.db['detections']
        
        # Create indexes for efficient querying
        self.collection.create_index([("plate_number", 1)])
        self.collection.create_index([("timestamp", -1)])
        
        # Initialize AWS S3
        print("Connecting to AWS S3...")
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=s3_region
        )
        self.s3_bucket = s3_bucket_name
        
        # Verify S3 bucket exists
        try:
            self.s3_client.head_bucket(Bucket=s3_bucket_name)
            print(f"S3 bucket '{s3_bucket_name}' verified")
        except ClientError:
            print(f"Warning: Cannot access S3 bucket '{s3_bucket_name}'")
        
        # Motion detection parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.min_contour_area = 5000
        
        # Initialize video capture
        self.cap = None
        
        print("Initialization complete")
        
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
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vehicle_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        return vehicle_contours
    
    def encode_image_to_base64(self, image):
        """Convert OpenCV image to base64 string for OpenAI API"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to JPEG in memory
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
    
    def extract_license_plate_openai(self, frame, contour):
        """
        Extract and read license plate using OpenAI Vision API
        
        Args:
            frame: Original frame
            contour: Contour of detected vehicle
            
        Returns:
            Tuple of (plate_text, confidence, plate_image)
        """
        # Get bounding box of vehicle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract vehicle region with padding
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return None, 0, None
        
        try:
            # Encode image for OpenAI
            image_base64 = self.encode_image_to_base64(vehicle_roi)
            
            # Call OpenAI Vision API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image and extract the license plate number if visible. 
                                Return ONLY the license plate text (letters and numbers), with no spaces or dashes. 
                                If no license plate is clearly visible, return 'NONE'. 
                                Be precise and only return valid license plate characters."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.0
            )
            
            plate_text = response.choices[0].message.content.strip().upper()
            
            # Validate the response
            if plate_text == "NONE" or len(plate_text) < 4 or len(plate_text) > 10:
                return None, 0, vehicle_roi
            
            # Remove any remaining spaces or punctuation
            plate_text = ''.join(c for c in plate_text if c.isalnum())
            
            # Check if it looks like a plate (has both letters and numbers)
            has_letter = any(c.isalpha() for c in plate_text)
            has_number = any(c.isdigit() for c in plate_text)
            
            if has_letter and has_number:
                # OpenAI doesn't provide confidence, so we'll use a fixed high value
                confidence = 0.95
                return plate_text, confidence, vehicle_roi
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
        
        return None, 0, vehicle_roi
    
    def upload_to_s3(self, image, plate_text, timestamp):
        """
        Upload image to S3 and return the S3 URL
        
        Args:
            image: OpenCV image
            plate_text: License plate text
            timestamp: Timestamp string
            
        Returns:
            S3 URL of uploaded image
        """
        # Generate S3 key (path)
        s3_key = f"plates/{timestamp[:8]}/{timestamp}_{plate_text}.jpg"
        
        try:
            # Convert image to JPEG in memory
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = buffer.tobytes()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=image_bytes,
                ContentType='image/jpeg',
                Metadata={
                    'plate_number': plate_text,
                    'timestamp': timestamp
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            return s3_url
            
        except ClientError as e:
            print(f"S3 upload error: {e}")
            return None
    
    def save_to_database(self, plate_text, confidence, s3_url, timestamp):
        """
        Save detection information to MongoDB
        
        Args:
            plate_text: License plate number
            confidence: Detection confidence
            s3_url: S3 URL of the image
            timestamp: Detection timestamp
        """
        document = {
            'plate_number': plate_text,
            'confidence': confidence,
            'image_url': s3_url,
            'timestamp': datetime.strptime(timestamp, "%Y%m%d_%H%M%S"),
            'created_at': datetime.now()
        }
        
        try:
            result = self.collection.insert_one(document)
            print(f"Saved to database: {plate_text} (ID: {result.inserted_id})")
            return result.inserted_id
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def query_plate_history(self, plate_number):
        """Query all detections for a specific plate number"""
        return list(self.collection.find({'plate_number': plate_number}).sort('timestamp', -1))
    
    def get_recent_detections(self, limit=10):
        """Get the most recent detections"""
        return list(self.collection.find().sort('timestamp', -1).limit(limit))
    
    def run(self, display=True):
        """
        Main detection loop
        
        Args:
            display: Whether to display video feed
        """
        self.connect_camera()
        
        print("Starting license plate detection... Press 'q' to quit")
        
        frame_skip = 3  # Process every 3rd frame for efficiency with API calls
        frame_count = 0
        detected_plates = {}  # Track recent detections
        
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
                    plate_text, confidence, plate_image = self.extract_license_plate_openai(
                        frame, contour
                    )
                    
                    if plate_text and confidence > 0.7:
                        current_time = datetime.now()
                        
                        # Check if recently detected (avoid duplicates within 30 seconds)
                        if plate_text not in detected_plates or \
                           (current_time - detected_plates[plate_text]).seconds > 30:
                            
                            # Generate timestamp
                            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                            
                            # Upload to S3
                            s3_url = self.upload_to_s3(plate_image, plate_text, timestamp)
                            
                            if s3_url:
                                # Save to database
                                self.save_to_database(plate_text, confidence, s3_url, timestamp)
                                detected_plates[plate_text] = current_time
                                
                                print(f"Detected: {plate_text} | Confidence: {confidence:.2f} | S3: {s3_url}")
                    
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
            self.mongo_client.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    CAMERA_URL = "rtsp://192.168.1.100/axis-media/media.amp"
    CAMERA_USERNAME = "your_username"
    CAMERA_PASSWORD = "your_password"
    
    OPENAI_API_KEY = "sk-your-openai-api-key"
    
    # MongoDB connection (local or cloud like MongoDB Atlas)
    MONGODB_URI = "mongodb://localhost:27017/"
    # For MongoDB Atlas: "mongodb+srv://username:password@cluster.mongodb.net/"
    
    # AWS credentials
    AWS_ACCESS_KEY = "your-aws-access-key"
    AWS_SECRET_KEY = "your-aws-secret-key"
    S3_BUCKET_NAME = "your-license-plate-bucket"
    S3_REGION = "us-east-1"
    
    # Create detector instance
    detector = LicensePlateDetector(
        camera_url=CAMERA_URL,
        username=CAMERA_USERNAME,
        password=CAMERA_PASSWORD,
        openai_api_key=OPENAI_API_KEY,
        mongodb_uri=MONGODB_URI,
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        s3_bucket_name=S3_BUCKET_NAME,
        s3_region=S3_REGION
    )
    
    # Run detection
    detector.run(display=True)
    
    # Example: Query plate history
    # history = detector.query_plate_history("ABC123")
    # for record in history:
    #     print(f"{record['timestamp']}: {record['plate_number']} - {record['image_url']}")