I can help you create a Python program for license plate detection from your security camera. This is a legitimate use case for monitoring your own property.I've created a comprehensive license plate detection system for your Axis camera. Here's what it does:

## Features:
1. **Motion Detection**: Uses background subtraction to detect moving vehicles
2. **License Plate Recognition**: Uses EasyOCR to read plate numbers
3. **Duplicate Prevention**: Avoids saving the same plate multiple times within 30 seconds
4. **Image & Log Storage**: Saves vehicle images and maintains a detection log
5. **Real-time Display**: Shows video feed with bounding boxes and detected plates

## Required Libraries:
Install these packages:
```bash
pip install opencv-python numpy easyocr requests
```

## Configuration:
1. **Find your camera's RTSP URL**: For Axis cameras, it's typically:
   - `rtsp://<camera_ip>/axis-media/media.amp`
   - Or check your camera's manual for the exact URL

2. **Update the script** with your camera's:
   - IP address
   - Username
   - Password

3. **Adjust parameters** as needed:
   - `min_contour_area`: Increase for larger vehicles only
   - `frame_skip`: Process fewer frames to reduce CPU load
   - Confidence threshold (currently 0.6)

## Output:
- Creates a `detected_plates` folder
- Saves images as: `timestamp_PLATENUM_confidence.jpg`
- Maintains `detections.log` with all detections

## Tips:
- Position your camera to capture plates at a good angle
- Ensure adequate lighting (IR illumination helps at night)
- Test during the day first to optimize settings
- Set `display=False` if running headless

The system works best with clear, frontal views of license plates. You may need to adjust the OCR filtering logic based on your region's plate format!