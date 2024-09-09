import cv2
import os
from ultralytics import YOLO

def process_video(video_path, output_dir):
    # Load YOLOv8 model (custom trained)
    model = YOLO('Models/zm.pt')
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Get video details
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0  # To keep track of frame number
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection on each frame
        results = model(frame, verbose=False)
        
        # Accessing the detections
        detections = results[0].boxes  # Use .boxes to access bounding boxes, confidences, and class IDs
        
        # Loop through detections
        for box in detections:
            # Extract bounding box coordinates, confidence, and class ID
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Convert box coordinates to integers
            confidence = box.conf.item()  # Confidence score
            cls = int(box.cls.item())  # Class ID
            
            # Check if bounding box is fully inside the image
            if xmin >= 0 and ymin >= 0 and xmax <= frame_width and ymax <= frame_height:
                
                # Crop the vehicle
                cropped_img = frame[ymin:ymax, xmin:xmax]
                
                # Save the cropped image
                class_name = model.names[cls]  # Get class name from index
                # Ensure the class_name directory exists
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)  # Create directory if it doesn't exist
                
                output_path = os.path.join(class_dir, f"{frame_count}_{xmin}_{ymin}.jpg")
                # print("saving img", output_path)
                cv2.imwrite(output_path, cropped_img)
        
        frame_count += 1  # Increment frame number
    
    # Release the video capture object
    print("Extracted all frames")
    cap.release()
    cv2.destroyAllWindows()

# Usage
# process_video('/workspace/zm.pt', '/workspace/dev.mp4', '/workspace/cropped/dev')
