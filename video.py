import cv2
import torch
import os
from models.posetrackingmodule import poseDetector

# Initialize YOLOv5 model for person detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Replace with 'yolov8' model path if you have it

def main():
    # Initialize pose detector
    pose_detector = poseDetector()

    # Open input video capture
    input_video = cv2.VideoCapture('junaid.mp4')
    
    # Get video properties
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output video writer
    output_path = os.path.join(output_dir, 'output(3).mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        # Read a frame from the input video
        ret, frame = input_video.read()
        
        if not ret:
            break
        
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Get detection results (bounding boxes)

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0:  # Class 0 in YOLO is 'person'
                
                # Step 2: Crop the detected person's bounding box region
                person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Step 3: Run pose detection on the cropped person ROI
                person_roi = pose_detector.findPose(person_roi)
                landmarks = pose_detector.findPosition(person_roi, draw=True)

                # Step 4: Map landmarks back to the main frame
                for lm in landmarks:
                    cx, cy = lm[1] + int(x1), lm[2] + int(y1)  # Adjust coordinates based on bounding box
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                # Print landmark information (optional)
                if landmarks:
                    print("Person landmarks:", landmarks)
        
        # Write the processed frame to the output video
        output_video.write(frame)
    
    # Release video capture and output video writer
    input_video.release()
    output_video.release()

if __name__ == "__main__":
    main()