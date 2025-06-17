import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, Canvas, Scrollbar
from PIL import Image, ImageTk
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 Model
model = YOLO("yolov8l-pose.pt").to("cuda")

# Define video sources
room_cameras = {
    "Room 1": r'room1_video.mp4',
    "Room 2": r'room2_video.mp4',
}

# Create a DataFrame for storing people count
data = pd.DataFrame(columns=["Room", "Timestamp", "People Count", "Status"])

# Initialize video capture and tracker
caps = {room: cv2.VideoCapture(url) for room, url in room_cameras.items()}
trackers = {room: DeepSort(max_age=50, n_init=5, max_iou_distance=0.3) for room in room_cameras.keys()}

# --------------------- GUI DESIGN ---------------------
window = tk.Tk()
window.title("AI-Based Multi-Room Monitoring System")
window.geometry("1600x1000")
window.configure(bg="black")

top_frame = tk.Frame(window, bg="black")
top_frame.pack(side="top", fill="x", pady=5)

status_label = tk.Label(top_frame, text="AI-Based Multi-Room Monitoring System", font=("Arial", 18, "bold"), fg="white", bg="black")
status_label.pack(pady=5)

def start_monitoring():
    process_frames()

def stop_monitoring():
    for cap in caps.values():
        cap.release()
    data.to_excel("People_Count_Report.xlsx", index=False)
    status_label.config(text="Monitoring Stopped", fg="red")
    window.quit()

def save_to_csv():
    data.to_csv("People_Count_Report.csv", index=False)
    status_label.config(text="📁 Report Saved as CSV!", fg="#00C853")

def exit_application():
    stop_monitoring()
    window.quit()

button_frame = tk.Frame(top_frame, bg="black")
button_frame.pack()

start_button = tk.Button(button_frame, text="START", command=start_monitoring, bg="#00C853", fg="white", font=("Arial", 12, "bold"), width=15)
start_button.grid(row=0, column=0, padx=10, pady=5)

stop_button = tk.Button(button_frame, text="STOP", command=stop_monitoring, bg="#D50000", fg="white", font=("Arial", 12, "bold"), width=15)
stop_button.grid(row=0, column=1, padx=10, pady=5)

save_csv_button = tk.Button(button_frame, text="SAVE TO CSV", command=save_to_csv, bg="#FF9100", fg="black", font=("Arial", 12, "bold"), width=15)
save_csv_button.grid(row=0, column=2, padx=10, pady=5)

exit_button = tk.Button(button_frame, text="EXIT", command=exit_application, bg="#9C27B0", fg="white", font=("Arial", 12, "bold"), width=15)
exit_button.grid(row=0, column=3, padx=10, pady=5)

video_container = tk.Frame(window, bg="black")
video_container.pack(side="left", padx=10, pady=10, fill="both", expand=True)

canvas_frame = Canvas(video_container, bg="black")
scroll_y = Scrollbar(video_container, orient="vertical", command=canvas_frame.yview)
scroll_y.pack(side="right", fill="y")

canvas_frame.configure(yscrollcommand=scroll_y.set)
canvas_frame.pack(side="left", fill="both", expand=True)

video_frame = tk.Frame(canvas_frame, bg="black")
canvas_frame.create_window((0, 0), window=video_frame, anchor="nw")

canvas_list = {}
for room in room_cameras.keys():
    room_label = tk.Label(video_frame, text=room, font=("Arial", 14, "bold"), fg="#FFD700", bg="black")
    room_label.pack()

    canvas = tk.Canvas(video_frame, width=800, height=600, bg="black", bd=2, relief="solid")
    canvas.pack(pady=5)
    canvas_list[room] = canvas

def update_scroll_region():
    video_frame.update_idletasks()
    canvas_frame.config(scrollregion=canvas_frame.bbox("all"))

video_frame.bind("<Configure>", lambda e: update_scroll_region())

report_frame = tk.Frame(window, bg="black")
report_frame.pack(side="bottom", fill="x", padx=10, pady=10)

report_label = tk.Label(report_frame, text="📊 Occupancy Report", font=("Arial", 14, "bold"), fg="white", bg="black")
report_label.pack(pady=5)

columns = ("Room", "Timestamp", "People Count", "Status")
report_tree = ttk.Treeview(report_frame, columns=columns, show="headings", height=10)
report_tree.pack(fill="both", expand=True)

for col in columns:
    report_tree.heading(col, text=col)
    report_tree.column(col, width=150)

def generate_report():
    report_tree.delete(*report_tree.get_children())  # Clear previous data
    for _, row in data.iterrows():
        report_tree.insert("", "end", values=(row["Room"], row["Timestamp"], row["People Count"], row["Status"]))

generate_button = tk.Button(report_frame, text="Generate Report", command=generate_report, bg="#2962FF", fg="white", font=("Arial", 12, "bold"))
generate_button.pack(fill="x", padx=5, pady=5)

save_button = tk.Button(report_frame, text="Save Report", command=save_to_csv, bg="#FF9100", fg="black", font=("Arial", 12, "bold"))
save_button.pack(fill="x", padx=5, pady=5)

def process_frames():
    global data
    confidence_threshold = 0.6  

    for room_name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((800, 600, 3), dtype=np.uint8)

        results = model(frame)
        detections = []

        for result in results:
            for box in result.boxes:
                conf = box.conf[0].cpu().numpy()
                if model.names[int(box.cls[0])] == "person" and conf >= confidence_threshold:
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, None))

        tracked_objects = trackers[room_name].update_tracks(detections, frame=frame)
        person_count = len([track for track in tracked_objects if track.is_confirmed()])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Active" if person_count > 0 else "Inactive"

        cv2.putText(frame, f"People: {person_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Timestamp: {timestamp}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)

        frame = cv2.resize(frame, (800, 600))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        canvas_list[room_name].create_image(0, 0, anchor=tk.NW, image=img)
        canvas_list[room_name].image = img

        new_entry = pd.DataFrame([[room_name, timestamp, person_count, status]], columns=data.columns)
        data = pd.concat([data, new_entry], ignore_index=True)

    generate_report()
    window.after(30, process_frames)

window.mainloop()
