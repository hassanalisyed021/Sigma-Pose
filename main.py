import cv2
import torch
from models.posetrackingmodule import poseDetector  # Import the poseDetector class from your pose tracking module

# Initialize YOLOv8 model for person detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Replace with 'yolov8' model path if you have it

def main():
    # Initialize pose detector
    pose_detector = poseDetector()
    
    # Open video capture
    cap = cv2.VideoCapture(0)    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Step 1: Run YOLO model to detect persons
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

        # Display the frame
        cv2.imshow("YOLO + MediaPipe Pose", frame)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
