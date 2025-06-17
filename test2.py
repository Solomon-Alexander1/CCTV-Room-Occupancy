import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, Scrollbar, Canvas, Frame
from PIL import Image, ImageTk
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading

# Load YOLOv8 Model
model = YOLO("yolov8n-pose.pt").to("cuda")

# Video sources for different rooms
room_cameras = {
    "Room 1": r'room1_video.mp4',
    "Room 2": r'room2_video.mp4',
}

# Initialize video capture and trackers for each room
caps = {room: cv2.VideoCapture(url) for room, url in room_cameras.items()}
trackers = {room: DeepSort(max_age=30, n_init=1, max_iou_distance=0.5) for room in room_cameras.keys()}

# DataFrame to store people count
data = pd.DataFrame(columns=["Room", "Timestamp", "People Count", "Status"])

# GUI Setup
window = tk.Tk()
window.title("AI-Based Multi-Room Monitoring System")
window.geometry("1600x900")
window.configure(bg="black")

# Title Label
status_label = tk.Label(window, text="AI-Based Multi-Room Monitoring System", font=("Arial", 18, "bold"), fg="white", bg="black")
status_label.pack(pady=5)

# Function to Start Monitoring
def start_monitoring():
    for room_name in room_cameras.keys():
        threading.Thread(target=process_frames, args=(room_name,), daemon=True).start()

# Function to Stop Monitoring
def stop_monitoring():
    for cap in caps.values():
        cap.release()
    data.to_excel("People_Count_Report.xlsx", index=False)
    status_label.config(text="Monitoring Stopped", fg="red")
    window.quit()

# Save Data as CSV
def save_to_csv():
    data.to_csv("People_Count_Report.csv", index=False)
    status_label.config(text="📁 Report Saved as CSV!", fg="#00C853")

# Exit Application
def exit_application():
    stop_monitoring()
    window.quit()

# Buttons Frame
button_frame = tk.Frame(window, bg="black")
button_frame.pack()

# Buttons
buttons = [
    ("START", start_monitoring, "#00C853"),
    ("STOP", stop_monitoring, "#D50000"),
    ("SAVE TO CSV", save_to_csv, "#FF9100"),
    ("EXIT", exit_application, "#9C27B0")
]

for i, (text, command, color) in enumerate(buttons):
    tk.Button(button_frame, text=text, command=command, bg=color, fg="white", font=("Arial", 12, "bold"), width=15).grid(row=0, column=i, padx=10, pady=5)

# Main Content Frame (For Videos + Report)
main_frame = tk.Frame(window, bg="black")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Scrollable Video Frame
video_frame_container = tk.Frame(main_frame, bg="black")
video_frame_container.pack(side="left", fill="both", expand=True)

canvas_scroll = tk.Canvas(video_frame_container, bg="black")
canvas_scroll.pack(side="left", fill="both", expand=True)

scrollbar = Scrollbar(video_frame_container, orient="vertical", command=canvas_scroll.yview)
scrollbar.pack(side="right", fill="y")

video_frame = tk.Frame(canvas_scroll, bg="black")
canvas_scroll.create_window((0, 0), window=video_frame, anchor="nw")
canvas_scroll.configure(yscrollcommand=scrollbar.set)

canvas_list = {}
for room in room_cameras.keys():
    tk.Label(video_frame, text=room, font=("Arial", 14, "bold"), fg="#FFD700", bg="black").pack(pady=5)
    canvas = tk.Canvas(video_frame, width=640, height=480, bg="black", bd=2, relief="solid")
    canvas.pack(pady=5)
    canvas_list[room] = canvas

# Function to update scrollbar
def update_scroll_region(event=None):
    canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

video_frame.bind("<Configure>", update_scroll_region)

# Report Frame (Right Side)
report_frame = tk.Frame(main_frame, bg="black")
report_frame.pack(side="right", fill="both", padx=10, pady=10)

tk.Label(report_frame, text="📊 Occupancy Report", font=("Arial", 14, "bold"), fg="white", bg="black").pack(pady=5)

# Scrollable TreeView (with Scroll Bar)
tree_frame = tk.Frame(report_frame)
tree_frame.pack(fill="both", expand=True)

scrollbar_report = Scrollbar(tree_frame, orient="vertical")
report_tree = ttk.Treeview(tree_frame, columns=("Room", "Timestamp", "People Count", "Status"), show="headings", height=20, yscrollcommand=scrollbar_report.set)
scrollbar_report.config(command=report_tree.yview)
scrollbar_report.pack(side="right", fill="y")

for col in ("Room", "Timestamp", "People Count", "Status"):
    report_tree.heading(col, text=col)
    report_tree.column(col, width=150)

report_tree.pack(fill="both", expand=True)

# Generate Report Function
def generate_report():
    report_tree.delete(*report_tree.get_children())
    for _, row in data.iterrows():
        report_tree.insert("", "end", values=(row["Room"], row["Timestamp"], row["People Count"], row["Status"]))

# Save Report Button
tk.Button(report_frame, text="Generate Report", command=generate_report, bg="#2962FF", fg="white", font=("Arial", 12, "bold")).pack(fill="x", padx=5, pady=5)

# Function to Process Frames for Each Room
def process_frames(room_name):
    global data
    cap = caps[room_name]
    tracker = trackers[room_name]
    frame_skip = 2  # Skip alternate frames for efficiency

    while cap.isOpened():
        for _ in range(frame_skip):
            cap.read()  # Skip frames

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video when it ends
            continue

        frame = cv2.resize(frame, (640, 480))

        # YOLOv8 Inference
        results = model(frame, verbose=False)

        # Object Detection
        detections = []
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0])
                if model.names[class_id] == "person" and conf >= 0.5:
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, None))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Tracking
        tracked_objects = tracker.update_tracks(detections, frame=frame)
        person_count = sum(track.is_confirmed() for track in tracked_objects)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Active" if person_count > 0 else "Inactive"

        # Display Information
        cv2.putText(frame, f"People: {person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Time: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert Frame to Image for Display
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        canvas_list[room_name].create_image(0, 0, anchor=tk.NW, image=img)
        canvas_list[room_name].image = img

        new_entry = pd.DataFrame([[room_name, timestamp, person_count, status]], columns=data.columns)
        data = pd.concat([data, new_entry], ignore_index=True)

        generate_report()
        time.sleep(0.01)

window.mainloop()
