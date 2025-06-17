import cv2
import pandas as pd
import time
import threading
from ultralytics import YOLO
from collections import deque

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use YOLOv8 Nano for fast performance

# Video file paths
video_files = {
    "Room 1": "room1_video.mp4",  # Replace with actual file path
    "Room 2": "room2_video.mp4"   # Replace with actual file path
}

# Initialize CSV
csv_file = "people_count_log.csv"
columns = ["Timestamp", "Room Number", "People Count"]
df = pd.DataFrame(columns=columns)
df.to_csv(csv_file, index=False)

# Last logged time for each room
last_logged_time = {"Room 1": 0, "Room 2": 0}

# Moving average buffer to stabilize count
count_buffer = {"Room 1": deque(maxlen=10), "Room 2": deque(maxlen=10)}

# Function to process videos
def process_video(room_name, video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original FPS
    frame_delay = max(1, int(1000 / fps))  # Ensure smooth real-time speed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection with higher confidence and IOU threshold
        results = model(frame, conf=0.4, iou=0.5)

        # Count people detected
        current_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) == 0)

        # Store count in buffer for smoothing
        count_buffer[room_name].append(current_count)

        # Get the most frequent value in buffer
        stabilized_count = max(set(count_buffer[room_name]), key=count_buffer[room_name].count)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is "Person"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display room name and count
        cv2.putText(frame, f"{room_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"People Count: {stabilized_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save data to CSV only once per 2 seconds
        current_time = time.time()
        if current_time - last_logged_time[room_name] >= 2:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            new_data = pd.DataFrame([[timestamp, room_name, stabilized_count]], columns=columns)
            new_data.to_csv(csv_file, mode='a', header=False, index=False)
            last_logged_time[room_name] = current_time  # Update last logged time

        # Show video feed with bounding boxes
        cv2.imshow(f"People Counting - {room_name}", frame)

        # Ensure smooth playback
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run both videos in parallel
thread1 = threading.Thread(target=process_video, args=("Room 1", video_files["Room 1"]))
thread2 = threading.Thread(target=process_video, args=("Room 2", video_files["Room 2"]))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

print(f"Data saved to {csv_file}")
